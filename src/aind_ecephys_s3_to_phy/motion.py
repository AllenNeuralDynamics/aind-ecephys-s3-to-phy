"""Load and apply KS4 motion correction from a sorted asset.

The AIND sorting pipeline stores per-probe motion estimates (displacement
fields computed by Kilosort 4) in the sorted asset under::

    preprocessed/motion/<recording_stem>/motion/

This module locates those estimates, loads them as SpikeInterface
:class:`~spikeinterface.core.Motion` objects, and wraps a recording in
an ``InterpolateMotionRecording`` so that the exported binary contains
drift-corrected traces.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import s3fs
from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.motion import Motion
from spikeinterface.sortingcomponents.motion.motion_interpolation import interpolate_motion

logger = logging.getLogger(__name__)

# Files that Motion.load expects in the motion directory.
_MOTION_MARKER = "spikeinterface_info.json"


def find_motion_path(
    sorted_uri: str,
    recording_name: str,
    storage_options: dict[str, Any] | None = None,
) -> str | None:
    """Locate the motion estimate for a recording in a sorted asset.

    Parameters
    ----------
    sorted_uri : str
        S3 URI or local path to the sorted asset root.
    recording_name : str
        Analyzer Zarr filename (e.g.
        ``"experiment1_...50720_recording1.zarr"``).  The ``.zarr`` suffix
        is stripped to form the subdirectory name.
    storage_options : dict or None
        fsspec storage options for S3 access.

    Returns
    -------
    str or None
        Path or S3 URI to the motion directory, or ``None`` if no motion
        estimate is found.
    """
    rec_stem = recording_name.removesuffix(".zarr")
    motion_dir = f"{sorted_uri.rstrip('/')}/preprocessed/motion/{rec_stem}/motion"

    if motion_dir.startswith("s3://"):
        opts = storage_options or {"anon": True}
        fs = s3fs.S3FileSystem(**opts)
        marker = f"{motion_dir.replace('s3://', '')}/{_MOTION_MARKER}"
        if fs.exists(marker):
            return motion_dir
        return None

    local = Path(motion_dir)
    if (local / _MOTION_MARKER).exists():
        return motion_dir
    return None


def load_motion(
    motion_path: str,
    storage_options: dict[str, Any] | None = None,
) -> Motion:
    """Load a :class:`~spikeinterface.core.Motion` object.

    For S3 paths the motion files are downloaded to a temporary directory
    first, since ``Motion.load`` requires local filesystem access.

    Parameters
    ----------
    motion_path : str
        Local path or S3 URI to the motion directory.
    storage_options : dict or None
        fsspec storage options for S3 access.

    Returns
    -------
    Motion
        The loaded motion estimate.
    """
    if motion_path.startswith("s3://"):
        return _load_motion_s3(motion_path, storage_options)
    return Motion.load(motion_path)


def apply_motion_correction(
    recording: BaseRecording,
    motion: Motion,
) -> BaseRecording:
    """Wrap *recording* in an ``InterpolateMotionRecording``.

    Uses ``border_mode="force_extrapolate"`` so that the output has the
    same number of channels as the input — no channels are dropped at
    probe edges.

    Parameters
    ----------
    recording : BaseRecording
        The preprocessed recording (must carry probe geometry).
    motion : Motion
        Motion estimate from :func:`load_motion`.

    Returns
    -------
    BaseRecording
        A lazy recording that applies drift correction on-the-fly.
    """
    return interpolate_motion(
        recording,
        motion,
        border_mode="force_extrapolate",
        spatial_interpolation_method="kriging",
    )


# --- Internal helpers ---


def _load_motion_s3(
    motion_path: str,
    storage_options: dict[str, Any] | None,
) -> Motion:
    """Download motion files from S3 to a temp dir and load."""
    opts = storage_options or {"anon": True}
    fs = s3fs.S3FileSystem(**opts)
    prefix = motion_path.replace("s3://", "")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # List and download all files in the motion directory
        for entry in fs.ls(prefix, detail=False):
            name = entry.split("/")[-1]
            fs.get(entry, str(tmp_path / name))
        return Motion.load(tmp_path)

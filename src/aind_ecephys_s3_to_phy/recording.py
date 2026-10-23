"""Reconstruct a SpikeInterface preprocessing chain from a SortingAnalyzer.

The SortingAnalyzer stores the full recording provenance (preprocessing chain)
as a nested dictionary in its Zarr store. This module extracts that dict,
substitutes the raw recording path at the leaf, and reconstructs the full
recording object including ChannelSlice steps that
``get_preprocessing_dict_from_analyzer`` intentionally skips.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from spikeinterface.core.base import _load_extractor_from_dict
from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.zarrextractors import super_zarr_open


def get_recording_dict(
    analyzer_path: str,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the recording provenance dict from a SortingAnalyzer Zarr.

    Parameters
    ----------
    analyzer_path : str
        Path or S3 URI to the SortingAnalyzer ``.zarr`` directory.
    storage_options : dict or None
        fsspec storage options (e.g. ``{"anon": True}``).

    Returns
    -------
    dict
        Nested dictionary representing the full preprocessing chain.

    Raises
    ------
    ValueError
        If the analyzer Zarr has no ``recording`` dataset.
    """
    storage_options = storage_options or {}
    zarr_root = super_zarr_open(analyzer_path, mode="r", storage_options=storage_options)
    rec_field = zarr_root.get("recording")
    if rec_field is None:
        raise ValueError(f"No recording dict found in analyzer: {analyzer_path}")
    result: dict[str, Any] = rec_field[0]
    return result


def fix_recording_dict(rec_dict: dict[str, Any], raw_path: str) -> dict[str, Any]:
    """Substitute the raw recording path in a recording provenance dict.

    Deep-copies the dict and replaces the ``folder_path`` of the innermost
    (leaf) extractor — typically a ``ZarrRecordingExtractor`` — with
    *raw_path*. Also clears ``relative_paths`` flags throughout the chain.

    Parameters
    ----------
    rec_dict : dict
        Recording provenance dict from :func:`get_recording_dict`.
    raw_path : str
        S3 URI or local path to the raw recording Zarr.

    Returns
    -------
    dict
        A new dict with the leaf path substituted.
    """
    rec_dict = copy.deepcopy(rec_dict)
    _fix_leaf_path(rec_dict, raw_path)
    _clear_relative_paths(rec_dict)
    return rec_dict


def reconstruct_recording(
    analyzer_path: str,
    raw_path: str,
    storage_options: dict[str, Any] | None = None,
) -> BaseRecording:
    """Reconstruct the full preprocessed recording from a SortingAnalyzer.

    Extracts the recording chain from the analyzer Zarr, substitutes the raw
    recording path, and instantiates the full SpikeInterface preprocessing
    pipeline (including ChannelSlice steps).

    Parameters
    ----------
    analyzer_path : str
        Path or S3 URI to the SortingAnalyzer ``.zarr`` directory.
    raw_path : str
        S3 URI or local path to the raw recording Zarr.
    storage_options : dict or None
        fsspec storage options for accessing the analyzer Zarr.

    Returns
    -------
    BaseRecording
        The reconstructed preprocessed recording.
    """
    rec_dict = get_recording_dict(analyzer_path, storage_options)
    fixed = fix_recording_dict(rec_dict, raw_path)
    return _load_extractor_from_dict(fixed)


def lazy_recording_dict(
    analyzer_path: str,
    raw_path: str,
    storage_options: dict[str, Any] | None = None,
    raw_storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a recording provenance dict that reads the raw data from *raw_path* when loaded.

    Parameters
    ----------
    analyzer_path : str
        Path or S3 URI to the SortingAnalyzer ``.zarr`` directory.
    raw_path : str
        S3 URI or local path to the raw recording Zarr.
    storage_options : dict or None
        fsspec storage options for accessing the analyzer Zarr.
    raw_storage_options : dict or None
        fsspec storage options stored with the raw leaf. Defaults to ``{"anon": True}`` for an S3
        *raw_path*, since a stale AWS profile otherwise fails before unsigned access is tried.

    Returns
    -------
    dict
        A JSON-serializable dict that ``spikeinterface.load`` turns into the preprocessed recording.
    """
    rec_dict = fix_recording_dict(get_recording_dict(analyzer_path, storage_options), raw_path)
    if raw_storage_options is None and raw_path.startswith("s3://"):
        raw_storage_options = {"anon": True}
    if raw_storage_options is not None:
        _leaf(rec_dict)["kwargs"]["storage_options"] = raw_storage_options
    return rec_dict


# --- Name matching ---


def match_analyzer_to_raw(analyzer_name: str) -> str:
    """Derive the raw recording Zarr name from an analyzer Zarr name.

    Analyzer names have a ``_recordingN`` suffix that raw names lack:

    - ``experiment1_...50213-1_recording1.zarr`` → ``experiment1_...50213-1.zarr``
    - ``foo.zarr`` → ``foo.zarr`` (no suffix, returned as-is)

    Parameters
    ----------
    analyzer_name : str
        Filename of the postprocessed SortingAnalyzer Zarr.

    Returns
    -------
    str
        Corresponding raw recording Zarr filename.
    """
    return re.sub(r"_recording\d+\.zarr$", ".zarr", analyzer_name)


# --- Internal helpers ---


def _fix_leaf_path(d: dict[str, Any], new_path: str) -> None:
    """Recursively find the leaf extractor and replace its ``folder_path``."""
    for key in ("recording", "parent_recording"):
        child = d.get("kwargs", {}).get(key)
        if isinstance(child, dict) and "class" in child:
            _fix_leaf_path(child, new_path)
            return
    # Leaf node — replace the path
    d["kwargs"]["folder_path"] = new_path
    d["relative_paths"] = False


def _leaf(d: dict[str, Any]) -> dict[str, Any]:
    """Return the innermost extractor dict of a recording chain."""
    for key in ("recording", "parent_recording"):
        child = d.get("kwargs", {}).get(key)
        if isinstance(child, dict) and "class" in child:
            return _leaf(child)
    return d


def _clear_relative_paths(d: dict[str, Any]) -> None:
    """Set ``relative_paths`` to False throughout the chain."""
    if "relative_paths" in d:
        d["relative_paths"] = False
    for key in ("recording", "parent_recording"):
        child = d.get("kwargs", {}).get(key)
        if isinstance(child, dict) and "class" in child:
            _clear_relative_paths(child)

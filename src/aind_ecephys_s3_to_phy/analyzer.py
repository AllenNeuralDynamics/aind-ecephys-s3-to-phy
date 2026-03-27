"""Load SortingAnalyzers and discover recordings in S3 assets."""

from __future__ import annotations

from typing import Any

import s3fs
import spikeinterface as si
from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.sortinganalyzer import SortingAnalyzer

from aind_ecephys_s3_to_phy.recording import match_analyzer_to_raw


def load_analyzer(
    analyzer_path: str,
    storage_options: dict[str, Any] | None = None,
) -> SortingAnalyzer:
    """Load a SortingAnalyzer and ensure the ``templates`` extension is available.

    Parameters
    ----------
    analyzer_path : str
        Path or S3 URI to the SortingAnalyzer ``.zarr`` directory.
    storage_options : dict or None
        fsspec storage options.

    Returns
    -------
    SortingAnalyzer
        The loaded analyzer with ``templates`` extension loaded.

    Raises
    ------
    ValueError
        If the ``templates`` extension is not available.
    """
    backend_options = {"storage_options": storage_options or {}}
    analyzer: SortingAnalyzer = si.load_sorting_analyzer(analyzer_path, backend_options=backend_options)

    # Extensions are lazy for remote paths — explicitly load templates
    if analyzer.get_extension("templates") is None:
        analyzer.load_extension("templates")
    if analyzer.get_extension("templates") is None:
        raise ValueError(f"Analyzer at {analyzer_path} has no 'templates' extension (required for phy export).")

    return analyzer


def attach_recording(analyzer: SortingAnalyzer, recording: BaseRecording) -> None:
    """Attach a preprocessed recording to a SortingAnalyzer.

    Validates that the recording attributes match the analyzer's expectations,
    then sets both the temporary and internal recording references. The
    internal ``_recording`` assignment is needed because ``export_to_phy``
    checks ``has_recording()`` which only inspects ``_recording``.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        The analyzer to attach the recording to.
    recording : BaseRecording
        The preprocessed recording (must match analyzer's channel_ids, etc.).

    Raises
    ------
    ValueError
        If the recording attributes don't match the analyzer.
    """
    # set_temporary_recording validates attributes and channel locations
    analyzer.set_temporary_recording(recording)
    # Work around: export_to_phy checks has_recording() which only looks at
    # _recording, not _temporary_recording.
    analyzer._recording = recording


# --- Discovery ---


def list_recordings(
    sorted_asset_uri: str,
    storage_options: dict[str, Any] | None = None,
) -> list[tuple[str, int]]:
    """List postprocessed SortingAnalyzer Zarrs in a sorted asset.

    Parameters
    ----------
    sorted_asset_uri : str
        S3 URI of the sorted asset (e.g. ``s3://bucket/sorted_asset``).
    storage_options : dict or None
        fsspec storage options (defaults to anonymous access).

    Returns
    -------
    list of (name, size_bytes)
        Analyzer Zarr names and their total sizes.
    """
    fs = _get_filesystem(storage_options)
    prefix = sorted_asset_uri.replace("s3://", "") + "/postprocessed/"
    return _list_zarrs(fs, prefix)


def list_raw_recordings(
    raw_asset_uri: str,
    storage_options: dict[str, Any] | None = None,
) -> list[tuple[str, int]]:
    """List compressed raw recording Zarrs in a raw asset.

    Filters to Neuropixels recordings only (excludes NI-DAQ).

    Parameters
    ----------
    raw_asset_uri : str
        S3 URI of the raw asset.
    storage_options : dict or None
        fsspec storage options.

    Returns
    -------
    list of (name, size_bytes)
        Raw recording Zarr names and their total sizes.
    """
    fs = _get_filesystem(storage_options)
    prefix = raw_asset_uri.replace("s3://", "") + "/ecephys/ecephys_compressed/"
    all_zarrs = _list_zarrs(fs, prefix)
    return [(name, size) for name, size in all_zarrs if "Neuropix" in name]


def discover_pairs(
    sorted_asset_uri: str,
    raw_asset_uri: str,
    storage_options: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Match postprocessed analyzers to their raw recordings.

    Parameters
    ----------
    sorted_asset_uri : str
        S3 URI of the sorted asset.
    raw_asset_uri : str
        S3 URI of the raw asset.
    storage_options : dict or None
        fsspec storage options.

    Returns
    -------
    list of (analyzer_name, raw_name)
        Matched pairs. Unmatched analyzers are excluded.
    """
    analyzers = list_recordings(sorted_asset_uri, storage_options)
    raw_recs = list_raw_recordings(raw_asset_uri, storage_options)
    raw_names = {name for name, _ in raw_recs}

    pairs: list[tuple[str, str]] = []
    for az_name, _ in analyzers:
        raw_candidate = match_analyzer_to_raw(az_name)
        if raw_candidate in raw_names:
            pairs.append((az_name, raw_candidate))

    return pairs


# --- Filtering ---


def filter_recordings(
    names: list[str],
    *,
    probe: str | None = None,
    experiment: str | None = None,
) -> list[str]:
    """Filter recording names by probe ID and/or experiment.

    Parameters
    ----------
    names : list of str
        Recording names (e.g. analyzer zarr filenames).
    probe : str or None
        If given, keep only names containing this substring (e.g. ``"46101"``).
    experiment : str or None
        If given, keep only names starting with this prefix (e.g.
        ``"experiment1"``).

    Returns
    -------
    list of str
        Filtered names, in original order.
    """
    result = names
    if experiment is not None:
        result = [n for n in result if n.startswith(experiment)]
    if probe is not None:
        result = [n for n in result if probe in n]
    return result


# --- Internal helpers ---


def _get_filesystem(storage_options: dict[str, Any] | None) -> s3fs.S3FileSystem:
    """Create an S3FileSystem with the given options."""
    opts = storage_options or {"anon": True}
    return s3fs.S3FileSystem(**opts)


def _list_zarrs(fs: s3fs.S3FileSystem, prefix: str) -> list[tuple[str, int]]:
    """List ``.zarr`` directories under a prefix with their sizes."""
    entries = fs.ls(prefix, detail=True)
    results: list[tuple[str, int]] = []
    for entry in entries:
        name = entry["name"].split("/")[-1]
        if name.endswith(".zarr"):
            size: int = fs.du(entry["name"])
            results.append((name, size))
    return results

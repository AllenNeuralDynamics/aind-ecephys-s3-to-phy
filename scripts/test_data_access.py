# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "spikeinterface[full,streaming_extractors]",
#     "mtscomp",
#     "s3fs",
#     "wavpack-numcodecs",
# ]
# ///
"""MWE: Test S3 streaming vs local-first data access for the ecephys-to-phy pipeline.

Benchmarks each pipeline step for a single recording to determine
whether to stream from S3 or pre-download.

Usage:
    uv run --script scripts/test_data_access.py
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import s3fs

SORTED_ASSET = "s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14"
RAW_ASSET = "s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56"
WORK_DIR = Path("mwe_workdir")

# Limit binary export to this many seconds (None = full recording).
# Full recordings can be 95+ GB as .dat — use a small value for MWE testing.
EXPORT_DURATION_S: float | None = 60.0

# Collect timings for the summary table
timings: dict[str, float] = {}


@contextmanager
def timed(label: str):
    """Context manager that prints and records wall-clock time."""
    print(f"  ⏱  {label} ... ", end="", flush=True)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"{elapsed:.1f}s")
    timings[label] = elapsed


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(fs: s3fs.S3FileSystem) -> tuple[str, str]:
    """List recordings in both assets, pick smallest, return (analyzer_name, raw_name)."""
    section("1. Discovery")

    sorted_prefix = SORTED_ASSET.replace("s3://", "") + "/postprocessed/"
    raw_prefix = RAW_ASSET.replace("s3://", "") + "/ecephys/ecephys_compressed/"

    print("\nPostprocessed analyzers:")
    sorted_entries = fs.ls(sorted_prefix, detail=True)
    analyzer_sizes: dict[str, int] = {}
    for entry in sorted_entries:
        name = entry["name"].split("/")[-1]
        if name.endswith(".zarr"):
            size = fs.du(entry["name"])
            analyzer_sizes[name] = size
            print(f"  {name}  ({size / 1e6:.1f} MB)")

    print("\nRaw compressed recordings:")
    raw_entries = fs.ls(raw_prefix, detail=True)
    raw_sizes: dict[str, int] = {}
    for entry in raw_entries:
        name = entry["name"].split("/")[-1]
        if name.endswith(".zarr"):
            size = fs.du(entry["name"])
            raw_sizes[name] = size
            print(f"  {name}  ({size / 1e6:.1f} MB)")

    # Match analyzer names to raw recording names.
    # Analyzer: "experiment1_Record Node 109#Neuropix-PXI-100.50198_recording1.zarr"
    # Raw:      "experiment1_Record Node 109#Neuropix-PXI-100.50198.zarr"
    # Strip the "_recordingN" suffix from the analyzer name to find the raw match.
    # Also filter out NI-DAQ entries (not neural probes).
    neural_raw = {k: v for k, v in raw_sizes.items() if "Neuropix" in k}

    matched_pairs: list[tuple[str, str, int, int]] = []
    for az_name, az_size in analyzer_sizes.items():
        # Strip _recordingN.zarr → .zarr to get raw name
        raw_candidate = re.sub(r"_recording\d+\.zarr$", ".zarr", az_name)
        if raw_candidate in neural_raw:
            matched_pairs.append((az_name, raw_candidate, az_size, neural_raw[raw_candidate]))

    print(f"\nMatched analyzer → raw pairs ({len(matched_pairs)}):")
    for az, raw, az_s, raw_s in sorted(matched_pairs, key=lambda x: x[3]):
        print(f"  {az} ({az_s / 1e6:.0f} MB) → {raw} ({raw_s / 1e6:.0f} MB)")

    # Pick the pair with the smallest raw recording (shortest experiment → fastest test)
    az_name, raw_name, az_s, raw_s = min(matched_pairs, key=lambda x: x[3])
    print(f"\nSelected (smallest raw):")
    print(f"  Analyzer: {az_name} ({az_s / 1e6:.1f} MB)")
    print(f"  Raw:      {raw_name} ({raw_s / 1e6:.1f} MB)")

    return az_name, raw_name


def test_load_analyzer(fs: s3fs.S3FileSystem, analyzer_name: str) -> tuple[Any, Any]:
    """Load SortingAnalyzer from S3 and locally, compare timings."""
    import spikeinterface as si

    section("2. Load SortingAnalyzer — S3 vs Local")

    analyzer_uri = f"{SORTED_ASSET}/postprocessed/{analyzer_name}"
    local_analyzer_path = WORK_DIR / "analyzer_local.zarr"

    # 2a: Load from S3
    with timed("Load SortingAnalyzer from S3"):
        analyzer_remote = si.load_sorting_analyzer(analyzer_uri)

    print(f"  Has recording: {analyzer_remote.has_recording()}")
    print(f"  Saved extensions: {analyzer_remote.get_saved_extension_names()}")

    # 2b: Load templates extension from S3
    with timed("Load 'templates' extension from S3"):
        analyzer_remote.load_extension("templates")

    templates = analyzer_remote.get_extension("templates")
    if templates is not None:
        print(f"  Templates shape: {templates.get_data().shape}")

    # 2c: Download analyzer zarr to local
    s3_path = analyzer_uri.replace("s3://", "")
    if local_analyzer_path.exists():
        shutil.rmtree(local_analyzer_path)
    with timed("Download analyzer zarr to local (s3fs.get)"):
        fs.get(s3_path, str(local_analyzer_path), recursive=True)

    # 2d: Load from local
    with timed("Load SortingAnalyzer from local"):
        analyzer_local = si.load_sorting_analyzer(local_analyzer_path)

    print(f"  Extensions loaded: {list(analyzer_local.extensions.keys())}")

    return analyzer_remote, analyzer_local


def test_load_raw_recording(fs: s3fs.S3FileSystem, raw_name: str) -> tuple[Any, Any, Path]:
    """Load raw recording from S3 and locally, compare timings.

    Returns (rec_remote, rec_local, local_raw_path).
    """
    import spikeinterface as si

    section("3. Load Raw Recording")

    raw_uri = f"{RAW_ASSET}/ecephys/ecephys_compressed/{raw_name}"
    local_raw_path = WORK_DIR / "raw_recording.zarr"

    # 3a: Load from S3
    with timed("Load raw recording from S3"):
        rec_remote = si.load(raw_uri)

    n_ch = rec_remote.get_num_channels()
    n_frames = rec_remote.get_num_frames()
    dur = rec_remote.get_total_duration()
    dt = rec_remote.get_dtype()
    est_dat_gb = n_frames * n_ch * dt.itemsize / 1e9
    print(f"  Channels: {n_ch}, Frames: {n_frames}, Duration: {dur:.1f}s")
    print(f"  Dtype: {dt}, Estimated .dat size: {est_dat_gb:.2f} GB")

    # 3b: Stream 1s of traces
    with timed("Stream 1s of traces from S3"):
        traces = rec_remote.get_traces(start_frame=0, end_frame=int(rec_remote.sampling_frequency))
    print(f"  Chunk shape: {traces.shape}, dtype: {traces.dtype}")

    # 3c: Download raw zarr to local
    s3_path = raw_uri.replace("s3://", "")
    if local_raw_path.exists():
        shutil.rmtree(local_raw_path)
    with timed("Download raw recording zarr to local (s3fs.get)"):
        fs.get(s3_path, str(local_raw_path), recursive=True)

    # 3d: Load from local
    with timed("Load raw recording from local"):
        rec_local = si.load(local_raw_path)

    # Verify local load works
    with timed("Read 1s of traces from local"):
        traces_local = rec_local.get_traces(start_frame=0, end_frame=int(rec_local.sampling_frequency))
    print(f"  Traces match: {np.array_equal(traces, traces_local)}")

    return rec_remote, rec_local, local_raw_path


def _get_recording_dict_from_analyzer(analyzer_uri: str) -> dict[str, Any]:
    """Extract the recording provenance dict from a remote SortingAnalyzer zarr."""
    from spikeinterface.core.zarrextractors import super_zarr_open

    zarr_root = super_zarr_open(analyzer_uri, mode="r", storage_options={})
    rec_field = zarr_root.get("recording")
    if rec_field is None:
        raise ValueError(f"No recording dict found in analyzer: {analyzer_uri}")
    return rec_field[0]  # type: ignore[no-any-return]


def _fix_recording_dict(rec_dict: dict[str, Any], raw_uri: str) -> dict[str, Any]:
    """Deep-copy the recording dict and replace the leaf ZarrRecordingExtractor path.

    The analyzer stores the full preprocessing chain as a nested dict. The
    innermost extractor is a ZarrRecordingExtractor pointing at a relative path
    that only worked on the original pipeline machine. We substitute it with the
    correct S3 URI (or local path) for the raw recording.
    """
    import copy

    rec_dict = copy.deepcopy(rec_dict)

    def _fix_leaf(d: dict[str, Any]) -> None:
        for key in ("recording", "parent_recording"):
            child = d.get("kwargs", {}).get(key)
            if isinstance(child, dict) and "class" in child:
                _fix_leaf(child)
                return
        # This is the leaf — replace the folder_path
        d["kwargs"]["folder_path"] = raw_uri
        d["relative_paths"] = False

    def _clear_relative(d: dict[str, Any]) -> None:
        if "relative_paths" in d:
            d["relative_paths"] = False
        for key in ("recording", "parent_recording"):
            child = d.get("kwargs", {}).get(key)
            if isinstance(child, dict) and "class" in child:
                _clear_relative(child)

    _fix_leaf(rec_dict)
    _clear_relative(rec_dict)
    return rec_dict


def _print_recording_chain(d: dict[str, Any], depth: int = 0) -> None:
    """Print the nested class chain of a recording dict."""
    indent = "  " * depth
    cls = d.get("class", "?").rsplit(".", 1)[-1]
    non_rec_kwargs = {
        k: v
        for k, v in d.get("kwargs", {}).items()
        if not (isinstance(v, dict) and "class" in v) and k not in ("recording", "parent_recording")
    }
    kwarg_summary = ", ".join(f"{k}={_short_repr(v)}" for k, v in non_rec_kwargs.items())
    print(f"  {indent}{cls}({kwarg_summary})")
    for key in ("recording", "parent_recording"):
        child = d.get("kwargs", {}).get(key)
        if isinstance(child, dict) and "class" in child:
            _print_recording_chain(child, depth + 1)


def _short_repr(v: Any) -> str:
    r = repr(v)
    return r if len(r) <= 80 else r[:77] + "..."


def test_preprocessing(
    analyzer_remote: Any,
    analyzer_local: Any,
    raw_name: str,
    local_raw_path: Path,
) -> tuple[Any, Any]:
    """Reconstruct the full preprocessing chain from the analyzer's recording dict.

    Instead of get_preprocessing_dict_from_analyzer (which skips ChannelSlice
    steps), we reconstruct the full chain from the stored dict and substitute
    the raw recording path at the leaf.
    """
    from spikeinterface.core.base import _load_extractor_from_dict

    section("4. Reconstruct Preprocessing (full chain from recording dict)")

    analyzer_uri = str(analyzer_remote.folder)
    raw_s3_uri = f"{RAW_ASSET}/ecephys/ecephys_compressed/{raw_name}"

    # Extract and print the chain
    with timed("Get recording dict from analyzer (S3)"):
        rec_dict = _get_recording_dict_from_analyzer(analyzer_uri)

    print("  Preprocessing chain:")
    _print_recording_chain(rec_dict)

    # Reconstruct with S3 raw path
    rec_dict_s3 = _fix_recording_dict(rec_dict, raw_s3_uri)
    with timed("Reconstruct preprocessed recording (S3 raw)"):
        pp_remote = _load_extractor_from_dict(rec_dict_s3)
    print(f"  Type: {type(pp_remote).__name__}, Channels: {pp_remote.get_num_channels()}")
    print(f"  Filtered: {pp_remote.is_filtered()}")

    # Reconstruct with local raw path
    rec_dict_local = _fix_recording_dict(rec_dict, str(local_raw_path))
    with timed("Reconstruct preprocessed recording (local raw)"):
        pp_local = _load_extractor_from_dict(rec_dict_local)
    print(f"  Type: {type(pp_local).__name__}, Channels: {pp_local.get_num_channels()}")

    # Correctness: stream 1s of preprocessed traces from both
    with timed("Stream 1s preprocessed traces (S3)"):
        pp_traces = pp_remote.get_traces(start_frame=0, end_frame=int(pp_remote.sampling_frequency))
    print(f"  Shape: {pp_traces.shape}, dtype: {pp_traces.dtype}")
    print(f"  Value range: [{pp_traces.min():.1f}, {pp_traces.max():.1f}]")

    with timed("Read 1s preprocessed traces (local)"):
        pp_traces_local = pp_local.get_traces(start_frame=0, end_frame=int(pp_local.sampling_frequency))
    print(f"  Traces match: {np.allclose(pp_traces, pp_traces_local)}")

    return pp_remote, pp_local


def attach_recording(analyzer: Any, preprocessed_rec: Any, label: str) -> bool:
    """Attach preprocessed recording to analyzer, return True if successful.

    Uses set_temporary_recording for validation, then also sets _recording
    because export_to_phy checks has_recording() which only looks at _recording,
    not _temporary_recording. This is a SpikeInterface limitation.
    """
    try:
        analyzer.set_temporary_recording(preprocessed_rec)
        # Work around: export_to_phy checks has_recording() which ignores
        # _temporary_recording. Set _recording directly after validation.
        analyzer._recording = preprocessed_rec
        print(f"  attach_recording ({label}): OK")
        return True
    except ValueError as e:
        print(f"  attach_recording ({label}): FAILED")
        print(f"    Error: {e}")
        print(f"    Analyzer rec_attributes: {analyzer.rec_attributes}")
        print(f"    Recording channels: {preprocessed_rec.get_num_channels()}")
        print(f"    Recording fs: {preprocessed_rec.sampling_frequency}")
        return False


def test_export_to_phy(
    analyzer_remote: Any,
    pp_remote: Any,
    pp_local: Any,
) -> Path | None:
    """Export to phy and benchmark binary writing S3 vs local."""
    from spikeinterface.core.recording_tools import write_binary_recording
    from spikeinterface.exporters import export_to_phy

    section("5. export_to_phy + binary write benchmark")

    phy_output = WORK_DIR / "phy_output"

    if not attach_recording(analyzer_remote, pp_remote, "remote"):
        print("  Skipping export — cannot attach recording to remote analyzer")
        return None

    # 5a: Export spike data (no binary copy — fast)
    if phy_output.exists():
        shutil.rmtree(phy_output)
    with timed("export_to_phy (spike data only, no binary)"):
        export_to_phy(
            analyzer_remote,
            phy_output,
            copy_binary=False,
            compute_pc_features=False,
            compute_amplitudes=False,
            use_relative_path=True,
        )

    print("  Output files:")
    for f in sorted(phy_output.iterdir()):
        print(f"    {f.name}: {f.stat().st_size / 1024:.1f} KB")

    # 5b: Benchmark binary writing — S3 vs local
    # Use EXPORT_DURATION_S to limit how much we write
    n_frames_remote = pp_remote.get_num_frames()
    if EXPORT_DURATION_S is not None:
        max_frames = int(pp_remote.sampling_frequency * EXPORT_DURATION_S)
        if max_frames < n_frames_remote:
            print(f"\n  Limiting binary write to {EXPORT_DURATION_S}s ({max_frames} frames of {n_frames_remote})")
            pp_remote_limited = pp_remote.frame_slice(start_frame=0, end_frame=max_frames)
            pp_local_limited = pp_local.frame_slice(start_frame=0, end_frame=max_frames)
        else:
            pp_remote_limited = pp_remote
            pp_local_limited = pp_local
    else:
        pp_remote_limited = pp_remote
        pp_local_limited = pp_local

    n_ch = pp_remote_limited.get_num_channels()
    n_frames = pp_remote_limited.get_num_frames()
    dtype = pp_remote_limited.get_dtype()
    dat_size_mb = n_frames * n_ch * dtype.itemsize / 1e6
    print(f"  Binary write: {n_ch} ch × {n_frames} frames × {dtype} = {dat_size_mb:.1f} MB")

    dat_s3 = phy_output / "recording_s3.dat"
    dat_local = phy_output / "recording_local.dat"

    with timed("write_binary_recording (S3 streaming)"):
        write_binary_recording(pp_remote_limited, file_paths=dat_s3, dtype=dtype, n_jobs=1, chunk_duration="1s")

    with timed("write_binary_recording (local)"):
        write_binary_recording(pp_local_limited, file_paths=dat_local, dtype=dtype, n_jobs=1, chunk_duration="1s")

    # Correctness
    h_s3 = file_sha256(dat_s3)
    h_local = file_sha256(dat_local)
    print(f"  Binary hashes match: {h_s3 == h_local}")
    print(f"  S3 .dat size: {dat_s3.stat().st_size / 1e6:.1f} MB")
    print(f"  Local .dat size: {dat_local.stat().st_size / 1e6:.1f} MB")

    # Rename local dat to recording.dat for mtscomp test and update params.py
    final_dat = phy_output / "recording.dat"
    dat_local.rename(final_dat)
    dat_s3.unlink()

    # Update params.py with correct dat_path
    params_path = phy_output / "params.py"
    params_text = params_path.read_text()
    params_text = params_text.replace("dat_path = None", f"dat_path = r'recording.dat'")
    params_path.write_text(params_text)

    return phy_output


def test_mtscomp(phy_output: Path) -> None:
    """Compress recording.dat to .cbin and verify."""
    import mtscomp

    section("6. mtscomp Compression")

    dat_path = phy_output / "recording.dat"
    cbin_path = phy_output / "recording.cbin"
    ch_path = phy_output / "recording.ch"

    if not dat_path.exists():
        print("  No recording.dat found — skipping compression test")
        return

    # Read params.py to get metadata
    params: dict[str, Any] = {}
    exec(open(phy_output / "params.py").read(), params)  # noqa: S102
    n_channels = params["n_channels_dat"]
    sample_rate = params["sample_rate"]
    dtype = params["dtype"]

    dat_size = dat_path.stat().st_size
    print(f"  .dat size: {dat_size / 1e6:.1f} MB")

    with timed("mtscomp compress"):
        mtscomp.compress(
            str(dat_path),
            out=str(cbin_path),
            outmeta=str(ch_path),
            sample_rate=sample_rate,
            n_channels=n_channels,
            dtype=dtype,
        )

    cbin_size = cbin_path.stat().st_size
    print(f"  .cbin size: {cbin_size / 1e6:.1f} MB")
    print(f"  Compression ratio: {dat_size / cbin_size:.2f}x")

    # Verify readability
    r = mtscomp.Reader()
    r.open(str(cbin_path), str(ch_path))
    data = r[0 : int(sample_rate)]
    print(f"  Read-back check: shape={data.shape}, dtype={data.dtype}")
    r.close()

    # Show what updated params.py would look like
    print(f"\n  Updated params.py dat_path would be: r'recording.cbin'")


def print_summary() -> None:
    section("7. Timing Summary")

    max_label = max(len(k) for k in timings)
    print(f"\n  {'Step':<{max_label}}  {'Time':>8}")
    print(f"  {'-' * max_label}  {'-' * 8}")
    for label, elapsed in timings.items():
        print(f"  {label:<{max_label}}  {elapsed:>7.1f}s")


def main() -> None:
    print(f"Working directory: {WORK_DIR.resolve()}")
    WORK_DIR.mkdir(exist_ok=True)

    fs = s3fs.S3FileSystem(anon=True)

    # 1. Discovery
    analyzer_name, raw_name = discover(fs)

    # 2. Load SortingAnalyzer
    analyzer_remote, analyzer_local = test_load_analyzer(fs, analyzer_name)

    # 3. Load raw recording
    rec_remote, rec_local, local_raw_path = test_load_raw_recording(fs, raw_name)

    # 4. Preprocessing (reconstruct full chain from analyzer's recording dict)
    pp_remote, pp_local = test_preprocessing(analyzer_remote, analyzer_local, raw_name, local_raw_path)

    # 5. export_to_phy + binary write benchmark
    phy_output = test_export_to_phy(analyzer_remote, pp_remote, pp_local)

    # 6. mtscomp
    if phy_output is not None:
        test_mtscomp(phy_output)

    # 7. Summary
    print_summary()


if __name__ == "__main__":
    main()

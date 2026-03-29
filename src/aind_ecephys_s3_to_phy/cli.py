"""CLI entry point for s3-to-phy.

Usage::

    s3-to-phy convert --sorted-uri s3://... --raw-uri s3://...
    s3-to-phy convert --sorted-uri s3://... --raw-uri s3://... --probe 46101
    s3-to-phy list --sorted-uri s3://...
    s3-to-phy extract-recording --analyzer-uri s3://... --raw-uri s3://... --output out.dat
    s3-to-phy compress --phy-dir ./phy_output
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from aind_ecephys_s3_to_phy.analyzer import (
    attach_recording,
    discover_pairs,
    filter_recordings,
    list_raw_recordings,
    list_recordings,
    load_analyzer,
)
from aind_ecephys_s3_to_phy.export import compress_binary, export_phy
from aind_ecephys_s3_to_phy.recording import match_analyzer_to_raw, reconstruct_recording


@dataclass
class Convert:
    """Full pipeline: export spike data, write binary traces, compress."""

    sorted_uri: str
    """S3 URI or local path to the sorted asset."""

    raw_uri: str
    """S3 URI or local path to the raw asset."""

    output_dir: Path = Path(".")
    """Base output directory."""

    recordings: list[str] = field(default_factory=list)
    """Recording names to export (default: all matched pairs)."""

    probe: str | None = None
    """Filter recordings by probe ID substring (e.g. 46101)."""

    experiment: str | None = None
    """Filter recordings by experiment prefix (e.g. experiment1)."""

    compress: bool = True
    """Compress recording.dat to .cbin."""

    skip_check: bool = False
    """Skip mtscomp compression verification."""

    n_jobs: int = 1
    """Parallel jobs for binary writing and PC extraction."""

    chunk_duration: str = "1s"
    """Chunk duration for binary writing."""


@dataclass
class List:
    """List available recordings in a sorted asset."""

    sorted_uri: str
    """S3 URI or local path to the sorted asset."""

    raw_uri: str | None = None
    """S3 URI or local path to the raw asset (shows matched pairs)."""


@dataclass
class ExtractRecording:
    """Write preprocessed recording to a flat binary file.

    Reconstructs the full preprocessing chain from a SortingAnalyzer
    (including channel selection) and writes the result as a flat binary.
    With --compress, produces a .cbin (mtscomp) file instead.
    """

    analyzer_uri: str
    """S3 URI or path to a postprocessed SortingAnalyzer .zarr."""

    raw_uri: str
    """S3 URI or path to the raw recording .zarr."""

    output: Path
    """Output file path (e.g. preprocessed.dat)."""

    compress: bool = False
    """Compress output to .cbin (mtscomp)."""

    dtype: str | None = None
    """Output dtype (default: same as recording, typically int16)."""

    n_jobs: int = 1
    """Parallel jobs for binary writing."""

    chunk_duration: str = "1s"
    """Chunk duration for binary writing."""


@dataclass
class Compress:
    """Compress an existing recording.dat to mtscomp .cbin format."""

    phy_dir: Path
    """Path to a phy output directory containing recording.dat."""

    skip_check: bool = False
    """Skip compression verification."""


Cmd = Annotated[
    Convert | List | ExtractRecording | Compress,
    tyro.conf.arg(prefix_name=False),
]


def main() -> None:
    """CLI entry point."""
    cmd = tyro.cli(Cmd, prog="s3-to-phy")  # type: ignore[call-overload]
    match cmd:
        case Convert():
            _run_convert(cmd)
        case List():
            _run_list(cmd)
        case ExtractRecording():
            _run_extract_recording(cmd)
        case Compress():
            _run_compress(cmd)


# --- Subcommand implementations ---


def _run_convert(cmd: Convert) -> None:
    """Run the full conversion pipeline."""
    recordings = _resolve_recordings(cmd)
    sorted_name = cmd.sorted_uri.rstrip("/").split("/")[-1]

    n_total = len(recordings)
    n_ok = 0
    n_fail = 0

    for i, rec_name in enumerate(recordings, 1):
        raw_name = match_analyzer_to_raw(rec_name)
        analyzer_uri = f"{cmd.sorted_uri.rstrip('/')}/postprocessed/{rec_name}"
        raw_uri = f"{cmd.raw_uri.rstrip('/')}/ecephys/ecephys_compressed/{raw_name}"
        rec_stem = rec_name.removesuffix(".zarr")
        phy_dir = cmd.output_dir / sorted_name / rec_stem

        print(f"\n[{i}/{n_total}] {rec_name}")
        print(f"  Output: {phy_dir}")

        try:
            _convert_one(cmd, analyzer_uri, raw_uri, phy_dir)
            n_ok += 1
            print(f"  Done: {phy_dir}")
        except KeyboardInterrupt:
            print("\n  Interrupted — cleaning up partial output...", file=sys.stderr)
            _cleanup_partial(phy_dir)
            sys.exit(130)
        except Exception as exc:
            n_fail += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

    print(f"\nFinished: {n_ok} succeeded, {n_fail} failed out of {n_total}.")
    if n_fail > 0:
        sys.exit(1)


def _resolve_recordings(cmd: Convert) -> list[str]:
    """Determine which recordings to export based on CLI flags."""
    recordings = cmd.recordings

    if not recordings:
        print("Discovering recordings...")
        pairs = discover_pairs(cmd.sorted_uri, cmd.raw_uri)
        if not pairs:
            print("No matching analyzer/raw recording pairs found.", file=sys.stderr)
            sys.exit(1)
        recordings = [az for az, _ in pairs]

    recordings = filter_recordings(recordings, probe=cmd.probe, experiment=cmd.experiment)

    if not recordings:
        print("No recordings match the given filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Selected {len(recordings)} recording(s):")
    for name in recordings:
        print(f"  {name}")

    return recordings


def _convert_one(cmd: Convert, analyzer_uri: str, raw_uri: str, phy_dir: Path) -> None:
    """Convert a single recording to phy."""
    print("  Loading analyzer...")
    analyzer = load_analyzer(analyzer_uri)

    print("  Reconstructing preprocessed recording...")
    recording = reconstruct_recording(analyzer_uri, raw_uri)

    print("  Attaching recording to analyzer...")
    attach_recording(analyzer, recording)

    print("  Exporting to phy...")
    export_phy(
        analyzer,
        recording,
        phy_dir,
        compress=cmd.compress,
        check_compression=not cmd.skip_check,
        n_jobs=cmd.n_jobs,
        chunk_duration=cmd.chunk_duration,
    )


def _run_list(cmd: List) -> None:
    """List recordings and optionally match to raw recordings."""
    analyzers = list_recordings(cmd.sorted_uri)

    if cmd.raw_uri is not None:
        raw_recs = list_raw_recordings(cmd.raw_uri)
        raw_names = {name for name, _ in raw_recs}
        raw_sizes = {name: size for name, size in raw_recs}

        print(f"{'Analyzer':<80} {'Size':>10}  {'Raw Match':<60} {'Raw Size':>10}")
        print("-" * 165)
        for az_name, az_size in sorted(analyzers):
            raw_candidate = match_analyzer_to_raw(az_name)
            if raw_candidate in raw_names:
                raw_s = raw_sizes[raw_candidate]
                print(f"{az_name:<80} {az_size / 1e6:>9.1f}M  {raw_candidate:<60} {raw_s / 1e6:>9.1f}M")
            else:
                print(f"{az_name:<80} {az_size / 1e6:>9.1f}M  (no match)")
    else:
        print(f"{'Analyzer':<80} {'Size':>10}")
        print("-" * 92)
        for az_name, az_size in sorted(analyzers):
            print(f"{az_name:<80} {az_size / 1e6:>9.1f}M")


def _run_extract_recording(cmd: ExtractRecording) -> None:
    """Reconstruct the preprocessed recording and write to binary."""
    import mtscomp
    from spikeinterface.core.recording_tools import write_binary_recording

    print("Reconstructing preprocessed recording...")
    print(f"  Analyzer: {cmd.analyzer_uri}")
    print(f"  Raw:      {cmd.raw_uri}")
    recording = reconstruct_recording(cmd.analyzer_uri, cmd.raw_uri)

    n_ch = recording.get_num_channels()
    n_frames = recording.get_num_frames()
    dtype = cmd.dtype or str(recording.get_dtype())
    sample_rate = recording.sampling_frequency
    est_size_gb = n_frames * n_ch * recording.get_dtype().itemsize / 1e9
    print(f"  Channels: {n_ch}, Frames: {n_frames}, Duration: {recording.get_total_duration():.1f}s")
    print(f"  Estimated raw size: {est_size_gb:.2f} GB")

    cmd.output.parent.mkdir(parents=True, exist_ok=True)

    if cmd.compress:
        dat_path = cmd.output.with_suffix(".dat")
        print(f"Writing intermediate {dat_path}...")
        write_binary_recording(
            recording,
            file_paths=dat_path,
            dtype=dtype,
            n_jobs=cmd.n_jobs,
            chunk_duration=cmd.chunk_duration,
            progress_bar=True,
            pool_engine="thread",
        )
        cbin_path = cmd.output.with_suffix(".cbin")
        ch_path = cmd.output.with_suffix(".ch")
        print(f"Compressing to {cbin_path}...")
        mtscomp.compress(
            str(dat_path),
            out=str(cbin_path),
            outmeta=str(ch_path),
            sample_rate=sample_rate,
            n_channels=n_ch,
            dtype=dtype,
        )
        dat_path.unlink()
        print(f"Done: {cbin_path} ({cbin_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"Writing to {cmd.output}...")
        write_binary_recording(
            recording,
            file_paths=cmd.output,
            dtype=dtype,
            n_jobs=cmd.n_jobs,
            chunk_duration=cmd.chunk_duration,
            progress_bar=True,
            pool_engine="thread",
        )
        print(f"Done: {cmd.output} ({cmd.output.stat().st_size / 1e9:.2f} GB)")


def _run_compress(cmd: Compress) -> None:
    """Compress recording.dat in the given phy directory."""
    print(f"Compressing {cmd.phy_dir / 'recording.dat'}...")
    compress_binary(cmd.phy_dir, check=not cmd.skip_check)
    print(f"Done: {cmd.phy_dir / 'recording.cbin'}")


def _cleanup_partial(phy_dir: Path) -> None:
    """Remove a partial phy output directory after interruption."""
    dat_path = phy_dir / "recording.dat"
    if dat_path.exists():
        dat_path.unlink()
        print(f"  Removed partial {dat_path}", file=sys.stderr)
    elif phy_dir.exists() and not any(phy_dir.iterdir()):
        shutil.rmtree(phy_dir)


if __name__ == "__main__":
    main()

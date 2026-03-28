"""CLI entry point using pydantic-settings with subcommands.

Usage::

    s3-to-phy convert --sorted_uri s3://... --raw_uri s3://...
    s3-to-phy convert --sorted_uri s3://... --raw_uri s3://... --probe 46101
    s3-to-phy list --sorted_uri s3://...
    s3-to-phy compress --phy_dir ./phy_output
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliSubCommand

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


class ConvertSettings(BaseSettings):
    """Full pipeline: export spike data, write binary traces, compress.

    Given a sorted asset and its corresponding raw asset on S3, produces
    a phy directory for each selected recording.
    """

    sorted_uri: str = Field(description="S3 URI of the sorted asset")
    raw_uri: str = Field(description="S3 URI of the raw asset")
    output_dir: Path = Field(default=Path("."), description="Base output directory")
    recordings: list[str] = Field(default_factory=list, description="Recording names to export (default: all)")
    probe: str | None = Field(default=None, description="Filter recordings by probe ID substring (e.g. 46101)")
    experiment: str | None = Field(
        default=None, description="Filter recordings by experiment prefix (e.g. experiment1)"
    )
    compress: bool = Field(default=True, description="Compress recording.dat to .cbin")
    skip_check: bool = Field(default=False, description="Skip mtscomp compression verification")
    n_jobs: int = Field(default=1, description="Parallel jobs for binary writing")
    chunk_duration: str = Field(default="1s", description="Chunk duration for binary writing")

    def cli_cmd(self) -> None:
        """Run the full conversion pipeline."""
        recordings = self._resolve_recordings()
        sorted_name = self.sorted_uri.rstrip("/").split("/")[-1]

        n_total = len(recordings)
        n_ok = 0
        n_fail = 0

        for i, rec_name in enumerate(recordings, 1):
            raw_name = match_analyzer_to_raw(rec_name)
            analyzer_uri = f"{self.sorted_uri.rstrip('/')}/postprocessed/{rec_name}"
            raw_uri = f"{self.raw_uri.rstrip('/')}/ecephys/ecephys_compressed/{raw_name}"
            rec_stem = rec_name.removesuffix(".zarr")
            phy_dir = self.output_dir / sorted_name / rec_stem

            print(f"\n[{i}/{n_total}] {rec_name}")
            print(f"  Output: {phy_dir}")

            try:
                self._convert_one(analyzer_uri, raw_uri, phy_dir)
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

    def _resolve_recordings(self) -> list[str]:
        """Determine which recordings to export based on CLI flags."""
        recordings = self.recordings

        if not recordings:
            print("Discovering recordings...")
            pairs = discover_pairs(self.sorted_uri, self.raw_uri)
            if not pairs:
                print("No matching analyzer/raw recording pairs found.", file=sys.stderr)
                sys.exit(1)
            recordings = [az for az, _ in pairs]

        recordings = filter_recordings(recordings, probe=self.probe, experiment=self.experiment)

        if not recordings:
            print("No recordings match the given filters.", file=sys.stderr)
            sys.exit(1)

        print(f"Selected {len(recordings)} recording(s):")
        for name in recordings:
            print(f"  {name}")

        return recordings

    def _convert_one(self, analyzer_uri: str, raw_uri: str, phy_dir: Path) -> None:
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
            compress=self.compress,
            check_compression=not self.skip_check,
            n_jobs=self.n_jobs,
            chunk_duration=self.chunk_duration,
        )


class ListSettings(BaseSettings):
    """List available recordings in a sorted asset."""

    sorted_uri: str = Field(description="S3 URI of the sorted asset")
    raw_uri: str | None = Field(default=None, description="S3 URI of the raw asset (shows matched pairs)")

    def cli_cmd(self) -> None:
        """List recordings and optionally match to raw recordings."""
        analyzers = list_recordings(self.sorted_uri)

        if self.raw_uri is not None:
            raw_recs = list_raw_recordings(self.raw_uri)
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


class ExtractRecordingSettings(BaseSettings):
    """Write the preprocessed recording to a binary file.

    Reconstructs the full preprocessing chain from a SortingAnalyzer
    (including channel selection) and writes the result as a flat binary.
    With ``--compress``, produces a ``.cbin`` (mtscomp) file instead.
    Useful for feeding preprocessed data into pipelines other than phy
    (e.g. mipmap builders).
    """

    analyzer_uri: str = Field(description="S3 URI or path to a postprocessed SortingAnalyzer .zarr")
    raw_uri: str = Field(description="S3 URI or path to the raw recording .zarr")
    output: Path = Field(description="Output file path (e.g. preprocessed.dat or preprocessed.cbin)")
    compress: bool = Field(default=False, description="Compress output to .cbin (mtscomp)")
    dtype: str | None = Field(default=None, description="Output dtype (default: same as recording, typically int16)")
    n_jobs: int = Field(default=1, description="Parallel jobs for binary writing")
    chunk_duration: str = Field(default="1s", description="Chunk duration for binary writing")

    def cli_cmd(self) -> None:
        """Reconstruct the preprocessed recording and write to binary."""
        import mtscomp
        from spikeinterface.core.recording_tools import write_binary_recording

        print("Reconstructing preprocessed recording...")
        print(f"  Analyzer: {self.analyzer_uri}")
        print(f"  Raw:      {self.raw_uri}")
        recording = reconstruct_recording(self.analyzer_uri, self.raw_uri)

        n_ch = recording.get_num_channels()
        n_frames = recording.get_num_frames()
        dtype = self.dtype or str(recording.get_dtype())
        sample_rate = recording.sampling_frequency
        est_size_gb = n_frames * n_ch * recording.get_dtype().itemsize / 1e9
        print(f"  Channels: {n_ch}, Frames: {n_frames}, Duration: {recording.get_total_duration():.1f}s")
        print(f"  Estimated raw size: {est_size_gb:.2f} GB")

        self.output.parent.mkdir(parents=True, exist_ok=True)

        if self.compress:
            # Write intermediate .dat, compress to .cbin, delete .dat
            dat_path = self.output.with_suffix(".dat")
            print(f"Writing intermediate {dat_path}...")
            write_binary_recording(
                recording,
                file_paths=dat_path,
                dtype=dtype,
                n_jobs=self.n_jobs,
                chunk_duration=self.chunk_duration,
                progress_bar=True,
                pool_engine="thread",
            )
            cbin_path = self.output.with_suffix(".cbin")
            ch_path = self.output.with_suffix(".ch")
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
            print(f"Writing to {self.output}...")
            write_binary_recording(
                recording,
                file_paths=self.output,
                dtype=dtype,
                n_jobs=self.n_jobs,
                chunk_duration=self.chunk_duration,
                progress_bar=True,
                pool_engine="thread",
            )
            print(f"Done: {self.output} ({self.output.stat().st_size / 1e9:.2f} GB)")


class CompressSettings(BaseSettings):
    """Compress an existing recording.dat to mtscomp .cbin format."""

    phy_dir: Path = Field(description="Path to a phy output directory containing recording.dat")
    check: bool = Field(default=True, description="Verify compression (slower but safe)")

    def cli_cmd(self) -> None:
        """Compress recording.dat in the given phy directory."""
        print(f"Compressing {self.phy_dir / 'recording.dat'}...")
        compress_binary(self.phy_dir, check=self.check)
        print(f"Done: {self.phy_dir / 'recording.cbin'}")


class S3ToPhy(BaseSettings):
    """Convert AIND ecephys sorted recordings from S3 into phy directories."""

    model_config = dict(cli_prog_name="s3-to-phy", cli_kebab_case=True, cli_implicit_flags=True, populate_by_name=True)

    convert: CliSubCommand[ConvertSettings]
    list: CliSubCommand[ListSettings]
    extract_recording: CliSubCommand[ExtractRecordingSettings] = Field(alias="extract-recording")
    compress: CliSubCommand[CompressSettings]

    def cli_cmd(self) -> None:
        """Dispatch to the active subcommand."""
        CliApp.run_subcommand(self)


def main() -> None:
    """CLI entry point."""
    CliApp.run(S3ToPhy, cli_args=sys.argv[1:])


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

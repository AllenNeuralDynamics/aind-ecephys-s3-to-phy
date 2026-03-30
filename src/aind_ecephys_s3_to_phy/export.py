"""Export a SortingAnalyzer to a phy directory with compressed traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mtscomp
import numpy as np
import pandas as pd
from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.recording_tools import write_binary_recording
from spikeinterface.core.sortinganalyzer import SortingAnalyzer
from spikeinterface.exporters import export_to_phy


def export_phy(
    analyzer: SortingAnalyzer,
    recording: BaseRecording,
    output_dir: str | Path,
    *,
    compress: bool = True,
    check_compression: bool = True,
    n_jobs: int = 1,
    chunk_duration: str = "1s",
) -> Path:
    """Export a SortingAnalyzer to a phy directory.

    Writes spike data via SpikeInterface's ``export_to_phy``, then streams
    the preprocessed recording to a flat binary and optionally compresses it
    with mtscomp.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        Analyzer with recording attached (see :func:`~.analyzer.attach_recording`).
    recording : BaseRecording
        The preprocessed recording to write as binary traces.
    output_dir : str or Path
        Directory to write phy files into.
    compress : bool
        If True (default), compress ``recording.dat`` to ``.cbin`` format.
    check_compression : bool
        If True (default), verify compression by decompressing and comparing.
    n_jobs : int
        Number of parallel jobs for binary writing.
    chunk_duration : str
        Chunk duration for binary writing (e.g. ``"1s"``).

    Returns
    -------
    Path
        The output directory.
    """
    output_dir = Path(output_dir)
    dat_path = output_dir / "recording.dat"
    cbin_path = output_dir / "recording.cbin"
    ch_path = output_dir / "recording.ch"

    # 1. Write the binary first (most expensive step, skip if already done).
    #    This gives us local data for PC extraction in step 2.
    if cbin_path.exists():
        print(f"  Skipping binary write — {cbin_path} already exists")
    elif dat_path.exists():
        print(f"  Skipping binary write — {dat_path} already exists")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_binary_recording(
            recording,
            file_paths=dat_path,
            dtype=recording.get_dtype(),
            n_jobs=n_jobs,
            chunk_duration=chunk_duration,
            progress_bar=True,
            pool_engine="thread",
        )

    # 2. Export spike data (templates, amplitudes, PCs, metrics, etc.).
    #    If the .dat exists, swap the analyzer's recording to a local
    #    BinaryRecordingExtractor so PC extraction reads from disk, not S3.
    if dat_path.exists():
        from spikeinterface.core import BinaryRecordingExtractor

        local_rec = BinaryRecordingExtractor(
            file_paths=dat_path,
            sampling_frequency=recording.sampling_frequency,
            num_channels=recording.get_num_channels(),
            dtype=recording.get_dtype(),
        )
        local_rec.set_channel_locations(recording.get_channel_locations())
        analyzer._recording = local_rec

    #    Stash binary files so export_to_phy can have a clean directory.
    #    Use try/finally to guarantee restoration — these files are expensive.
    _binary_files = [cbin_path, ch_path, dat_path]
    _preserved: list[tuple[Path, Path]] = []
    for f in _binary_files:
        if f.exists():
            stash = output_dir.parent / f"._{f.name}.stash"
            if stash.exists():
                raise RuntimeError(
                    f"Stash file {stash} already exists from a previous interrupted run. "
                    f"Restore it manually to {f} before re-running."
                )
            f.rename(stash)
            _preserved.append((stash, f))

    try:
        export_to_phy(
            analyzer,
            output_dir,
            copy_binary=False,
            compute_pc_features=True,
            compute_amplitudes=True,
            use_relative_path=True,
            remove_if_exists=True,
            n_jobs=n_jobs,
            pool_engine="thread",
        )
    finally:
        # Restore binary files even if export_to_phy fails
        for stash, orig in _preserved:
            if stash.exists():
                orig.parent.mkdir(parents=True, exist_ok=True)
                stash.rename(orig)

    # Rewrite cluster_group.tsv using sorting labels
    _write_cluster_labels(analyzer, output_dir)

    # 3. Update params.py and optionally compress to .cbin
    if cbin_path.exists():
        _update_params_dat_path(output_dir, "recording.cbin")
    elif compress and dat_path.exists():
        _update_params_dat_path(output_dir, "recording.dat")
        compress_binary(output_dir, check=check_compression)
    elif dat_path.exists():
        _update_params_dat_path(output_dir, "recording.dat")

    return output_dir


def compress_binary(phy_dir: str | Path, *, check: bool = True) -> None:
    """Compress ``recording.dat`` to mtscomp ``.cbin`` format.

    Reads recording metadata from ``params.py``, compresses the ``.dat``
    file, updates ``params.py`` to point at the ``.cbin``, and deletes
    the original ``.dat``.

    Parameters
    ----------
    phy_dir : str or Path
        Path to a phy output directory containing ``recording.dat`` and
        ``params.py``.
    check : bool
        If True (default), verify the compressed file by decompressing and
        comparing against the original. Doubles I/O time but guarantees
        lossless compression.

    Raises
    ------
    FileNotFoundError
        If ``recording.dat`` or ``params.py`` is missing.
    """
    phy_dir = Path(phy_dir)
    dat_path = phy_dir / "recording.dat"
    cbin_path = phy_dir / "recording.cbin"
    ch_path = phy_dir / "recording.ch"

    if not dat_path.exists():
        raise FileNotFoundError(f"No recording.dat found in {phy_dir}")

    params = _read_params(phy_dir)

    mtscomp.compress(
        str(dat_path),
        out=str(cbin_path),
        outmeta=str(ch_path),
        sample_rate=params["sample_rate"],
        n_channels=params["n_channels_dat"],
        dtype=params["dtype"],
        check_after_compress=check,
    )

    _update_params_dat_path(phy_dir, "recording.cbin")
    dat_path.unlink()


# --- Internal helpers ---


def _read_params(phy_dir: Path) -> dict[str, Any]:
    """Parse ``params.py`` in a phy directory."""
    params_path = phy_dir / "params.py"
    if not params_path.exists():
        raise FileNotFoundError(f"No params.py found in {phy_dir}")
    namespace: dict[str, Any] = {"r": str, "np": np}
    exec(params_path.read_text(), namespace)
    return namespace


def _write_cluster_labels(analyzer: SortingAnalyzer, output_dir: Path) -> None:
    """Rewrite ``cluster_group.tsv`` using sorting labels and write individual label columns.

    Assigns phy cluster groups based on a combined heuristic:

    - **good**: KSLabel ``good`` AND decoder_label ``sua``
    - **mua**: everything else that isn't noise
    - **noise**: decoder_label ``noise`` OR default_qc ``False``

    Also writes each available label (KSLabel, decoder_label, default_qc,
    decoder_probability) as a separate ``cluster_*.tsv`` for visibility
    in phy's cluster view.
    """
    sorting = analyzer.sorting
    unit_ids = sorting.unit_ids
    n_units = len(unit_ids)
    cluster_ids = list(range(n_units))

    # Determine combined cluster group
    ks_label = sorting.get_property("KSLabel")
    decoder_label = sorting.get_property("decoder_label")
    default_qc = sorting.get_property("default_qc")

    groups: list[str] = ["unsorted"] * n_units
    for i in range(n_units):
        is_noise = False
        if decoder_label is not None and decoder_label[i] == "noise":
            is_noise = True
        if default_qc is not None and not default_qc[i]:
            is_noise = True

        if is_noise:
            groups[i] = "noise"
        elif ks_label is not None and ks_label[i] == "good" and decoder_label is not None and decoder_label[i] == "sua":
            groups[i] = "good"
        else:
            groups[i] = "mua"

    # Overwrite cluster_group.tsv
    cluster_group = pd.DataFrame({"cluster_id": cluster_ids, "group": groups})
    cluster_group.to_csv(output_dir / "cluster_group.tsv", sep="\t", index=False)

    # Write individual label columns as cluster_*.tsv
    _label_properties = ["KSLabel", "decoder_label", "default_qc", "decoder_probability"]
    for prop_name in _label_properties:
        values = sorting.get_property(prop_name)
        if values is not None:
            df = pd.DataFrame({"cluster_id": cluster_ids, prop_name: values})
            df.to_csv(output_dir / f"cluster_{prop_name}.tsv", sep="\t", index=False)


def _update_params_dat_path(phy_dir: Path, dat_filename: str) -> None:
    """Rewrite the ``dat_path`` line in ``params.py``."""
    params_path = phy_dir / "params.py"
    lines = params_path.read_text().splitlines(keepends=True)
    new_lines: list[str] = []
    for line in lines:
        if line.startswith("dat_path"):
            new_lines.append(f"dat_path = r'{dat_filename}'\n")
        else:
            new_lines.append(line)
    params_path.write_text("".join(new_lines))

"""Export a SortingAnalyzer to a phy directory with compressed traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mtscomp
import numpy as np
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

    # 1. Export spike data (always — fast, ensures latest extensions).
    #    export_to_phy requires an empty directory and remove_if_exists
    #    does shutil.rmtree, so stash binary files in the parent directory.
    _binary_files = [cbin_path, ch_path, dat_path]
    _preserved: list[tuple[Path, Path]] = []
    for f in _binary_files:
        if f.exists():
            tmp = output_dir.parent / f"._{f.name}.stash"
            f.rename(tmp)
            _preserved.append((tmp, f))

    export_to_phy(
        analyzer,
        output_dir,
        copy_binary=False,
        compute_pc_features=True,
        compute_amplitudes=True,
        use_relative_path=True,
        remove_if_exists=True,
    )

    # Restore binary files
    for tmp, orig in _preserved:
        tmp.rename(orig)

    # 2. Write the preprocessed recording as flat binary (skip if already done)
    if cbin_path.exists():
        print(f"  Skipping binary write — {cbin_path} already exists")
    elif dat_path.exists():
        print(f"  Skipping binary write — {dat_path} already exists")
    else:
        write_binary_recording(
            recording,
            file_paths=dat_path,
            dtype=recording.get_dtype(),
            n_jobs=n_jobs,
            chunk_duration=chunk_duration,
            progress_bar=True,
            pool_engine="thread",
        )

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

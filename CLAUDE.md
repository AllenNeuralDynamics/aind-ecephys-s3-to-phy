# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aind-ecephys-s3-to-phy** — CLI tools to convert AIND ecephys sorted recordings and raw recordings (two separate S3 assets) into [phy](https://github.com/cortex-lab/phy) directories for manual curation, including raw trace access. Uses [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) for the conversion and [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for CLI argument parsing.

`src/` layout, `hatchling` build backend, `uv` for dependency management.

## Commands

```bash
# Setup
uv sync

# Full linting + tests (use before committing)
./scripts/run_linters_and_checks.sh -c

# Format only (no checks)
./scripts/run_linters_and_checks.sh

# Individual tools (when venv is active, drop "uv run --frozen")
uv run --frozen ruff format
uv run --frozen ruff check
uv run --frozen mypy
uv run --frozen interrogate -v
uv run --frozen codespell --check-filenames
uv run --frozen pytest --cov aind_ecephys_s3_to_phy

# Run a single test
uv run --frozen pytest tests/test_example.py -k "test_name"

# Pass extra pytest args via the script
./scripts/run_linters_and_checks.sh -c -- -k "test_name"
```

## Code Style & Tooling

- **Line length**: 120
- **Target Python**: 3.11+ (minimum 3.10)
- **Ruff rules**: Q, RUF100, C90, I, F, E, W, D, UP, ANN, PYI (type annotations not required in tests)
- **Docstrings**: NumPy convention, 30% coverage minimum (interrogate)
- **Type checking**: mypy strict mode on `src/` (relaxed in tests)
- **Commits**: Angular convention via commitizen (semantic versioning)

## Architecture

### Domain: AIND Ecephys Data on S3

Two separate S3 assets are involved per session:

1. **Raw asset** (e.g. `s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56/`)
   - Contains compressed Zarr recordings under `ecephys/ecephys_compressed/`
   - One Zarr per probe per experiment (e.g. `experiment1_Record Node 109#Neuropix-PXI-100.46101.zarr`)

2. **Sorted asset** (e.g. `s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14/`)
   - `postprocessed/` — SortingAnalyzer as Zarr (one per recording), with precomputed extensions (templates, waveforms, PCs, quality metrics, etc.)
   - `preprocessed/` — JSON files describing SpikeInterface preprocessing chains (not actual trace data)
   - `spikesorted/` — NumpySorting format
   - `curated/` — Subset with auto-curation labels

Each sorted asset can have many recordings (e.g. 10 probes x 3 experiments = 30). Users select specific probes/streams/experiments to export.

### Conversion Pipeline

```
SortingAnalyzer (S3)
  ├─ Load from S3 (stream, don't download)
  ├─ Extract recording dict (full preprocessing chain with ChannelSlice)
  └─ Load templates extension (lazy, on demand)
         │
         ▼
Recording dict  ──→  Substitute raw Zarr S3 URI at leaf
                     ──→  _load_extractor_from_dict()
                          ──→  Preprocessed recording (373 ch, filtered)
                               │
                               ├─ Attach to analyzer via _recording = rec
                               │
                               ▼
                     si.export_to_phy(analyzer, copy_binary=False)
                          ──→  spike data (.npy, .tsv, params.py)
                               │
                               ▼
                     write_binary_recording(rec, "recording.dat")
                          ──→  Streams from S3 → flat .dat
                               │
                               ▼
                     mtscomp.compress("recording.dat")
                          ──→  recording.cbin + recording.ch
                               ──→  Update params.py, delete .dat
```

Key details:
- **Preprocessing reconstruction**: `get_preprocessing_dict_from_analyzer()` extracts the chain from the SortingAnalyzer Zarr. `apply_preprocessing_pipeline()` applies it to the raw recording, so phy shows filtered/referenced traces (not raw).
- **mtscomp (.cbin)**: Phy natively supports `.cbin` files via phylib's `MtscompEphysReader`. ~3x lossless compression with random-access decompression. Avoids storing massive flat `.dat` files.
- **Phy trace format support**: `.dat`/`.bin`/`.raw`/`.mda` (flat binary), `.cbin` (mtscomp), `.npy` (numpy). No Zarr/HDF5.
- **`export_to_phy` requires**: SortingAnalyzer with at least `templates` extension computed. Single-segment recordings only.
- **S3 access**: SpikeInterface uses fsspec/s3fs. Anonymous access tried first (works for public `aind-open-data`). Standard AWS credential chain for private buckets.

### MWE Results & Design Decisions (from `scripts/test_data_access.py`)

**Data access strategy** (benchmarked on a single recording, 60s binary write):
- **SortingAnalyzer**: Stream from S3 (3s load + 2.4s templates). Do NOT download first (55s download + 9s load is much slower). Extensions are lazy-loaded on demand for remote paths.
- **Raw recording → binary**: Stream directly from S3 during `write_binary_recording` (87s for 60s of data). Only 1.35x slower than local, and avoids the ~290s download of 30 GB raw zarr. **Streaming is the right default.**
- **mtscomp**: 2.52x compression ratio, 10s for 1.3 GB. Fast and worth doing by default.

**Critical gotchas discovered**:
- **`get_preprocessing_dict_from_analyzer()` skips `ChannelSlice` steps** (by design). The AIND pipeline applies bad-channel removal via `ChannelSliceRecording`, so the simplified pipeline produces a recording with 384 channels while the analyzer expects ~373. **Must reconstruct the full chain** from the analyzer's stored recording dict (`zarr_root["recording"][0]`) and substitute the raw recording path at the leaf `ZarrRecordingExtractor`.
- **`export_to_phy` ignores `_temporary_recording`**: The function checks `has_recording()` which only looks at `_recording`, not `_temporary_recording`. Workaround: set `analyzer._recording = preprocessed_rec` directly after validation via `set_temporary_recording()`.
- **Raw recordings use WavPack codec**: Requires `wavpack-numcodecs` package.
- **Analyzer naming**: Postprocessed zarr names have `_recording1` suffix (e.g. `...50213-1_recording1.zarr`) while raw zarr names don't (e.g. `...50213-1.zarr`). Strip `_recording\d+` to match.
- **Only experiment1 has postprocessed data** in the test dataset, and raw recordings are ~30 GB each (95 GB as flat .dat).

### CLI Design (pydantic-settings)

Uses `pydantic_settings.BaseSettings` with `CliApp.run` pattern and subcommands:

- **Happy-path command**: One command that downloads + preprocesses + exports + compresses
- **Subcommands**: Separate steps for download, export, compress — for flexibility/debugging
- **Entry point**: `[project.scripts]` in pyproject.toml

Output structure: working directory based on S3 asset name, with phy folders per probe/experiment.

### Key Dependencies

- **spikeinterface** (PyPI) — SortingAnalyzer loading, preprocessing pipeline, `export_to_phy`
- **pydantic-settings** — CLI argument parsing via BaseSettings
- **mtscomp** — Compress recording.dat → .cbin for phy trace view
- **s3fs/fsspec** — S3 access (transitive via spikeinterface)

### Local Reference Repos

- SpikeInterface: `~/Documents/Code/spikeinterface`
- phy: `~/Documents/Code/phy`

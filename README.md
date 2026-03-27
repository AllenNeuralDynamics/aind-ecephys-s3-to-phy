# aind-ecephys-s3-to-phy

![CI](https://github.com/AllenNeuralDynamics/aind-ecephys-s3-to-phy/actions/workflows/ci-call.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-ecephys-s3-to-phy)](https://pypi.org/project/aind-ecephys-s3-to-phy/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border.json)](https://github.com/copier-org/copier)

Convert AIND ecephys sorted recordings from S3 into [phy](https://github.com/cortex-lab/phy) directories for manual curation, with raw trace access via compressed binary ([mtscomp](https://github.com/int-brain-lab/mtscomp)).

## Installation

```bash
pip install aind-ecephys-s3-to-phy
```

## Usage

The package provides the `s3-to-phy` CLI. Two S3 assets are needed: the **sorted asset** (containing postprocessed SortingAnalyzer Zarrs) and the **raw asset** (containing compressed raw recordings).

### List available recordings

```bash
s3-to-phy list --sorted_uri s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14
```

With matched raw recordings:

```bash
s3-to-phy list \
  --sorted_uri s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14 \
  --raw_uri s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56
```

### Convert recordings to phy

Convert all recordings:

```bash
s3-to-phy convert \
  --sorted_uri s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14 \
  --raw_uri s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56
```

Filter by probe or experiment:

```bash
s3-to-phy convert \
  --sorted_uri s3://... --raw_uri s3://... \
  --probe 46101

s3-to-phy convert \
  --sorted_uri s3://... --raw_uri s3://... \
  --experiment experiment1
```

Convert a specific recording:

```bash
s3-to-phy convert \
  --sorted_uri s3://... --raw_uri s3://... \
  --recordings '["experiment1_Record Node 109#Neuropix-PXI-100.46101_recording1.zarr"]'
```

### Compress an existing phy directory

If you have a phy directory with an uncompressed `recording.dat`:

```bash
s3-to-phy compress --phy_dir ./phy_output
```

### Python API

```python
from aind_ecephys_s3_to_phy.analyzer import load_analyzer, attach_recording, discover_pairs
from aind_ecephys_s3_to_phy.recording import reconstruct_recording
from aind_ecephys_s3_to_phy.export import export_phy

sorted_uri = "s3://aind-open-data/ecephys_786867_..._sorted_..."
raw_uri = "s3://aind-open-data/ecephys_786867_..."

# Discover available recording pairs
pairs = discover_pairs(sorted_uri, raw_uri)

# Convert a single recording
analyzer_name, raw_name = pairs[0]
analyzer_zarr = f"{sorted_uri}/postprocessed/{analyzer_name}"
raw_zarr = f"{raw_uri}/ecephys/ecephys_compressed/{raw_name}"

analyzer = load_analyzer(analyzer_zarr)
recording = reconstruct_recording(analyzer_zarr, raw_zarr)
attach_recording(analyzer, recording)

export_phy(analyzer, recording, "./phy_output", compress=True)
```

### S3 credentials

Public buckets (like `aind-open-data`) work without credentials. For private buckets, configure AWS credentials via the standard methods (environment variables, `~/.aws/credentials`, IAM roles). SpikeInterface uses `s3fs`/`fsspec` which respects the standard AWS credential chain.

## Development

```bash
uv sync
```

Run the full linting and testing suite:

```bash
./scripts/run_linters_and_checks.sh -c
```

Or run individual commands:

```bash
uv run --frozen ruff format          # Code formatting
uv run --frozen ruff check           # Linting
uv run --frozen mypy                 # Type checking
uv run --frozen interrogate -v       # Documentation coverage
uv run --frozen codespell --check-filenames  # Spell checking
uv run --frozen pytest --cov aind_ecephys_s3_to_phy # Tests with coverage
```

Run integration tests (requires S3 access):

```bash
uv run --frozen pytest -m integration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

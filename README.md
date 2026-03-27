# aind-ecephys-s3-to-phy

![CI](https://github.com/AllenNeuralDynamics/aind-ecephys-s3-to-phy/actions/workflows/ci-call.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-ecephys-s3-to-phy)](https://pypi.org/project/aind-ecephys-s3-to-phy/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border.json)](https://github.com/copier-org/copier)

Convert AIND ecephys sorted recordings from S3 into [phy](https://github.com/cortex-lab/phy) directories for manual curation, with raw trace access via compressed binary ([mtscomp](https://github.com/int-brain-lab/mtscomp)).

Two S3 assets are needed per session: a **sorted asset** (postprocessed SortingAnalyzer Zarrs) and its corresponding **raw asset** (compressed Zarr recordings). The tool reconstructs the full SpikeInterface preprocessing chain — including bad-channel removal — from the analyzer, streams traces from S3, and compresses them to `.cbin` for phy's trace viewer.

## Installation

```bash
pip install aind-ecephys-s3-to-phy
```

## Quick start

```bash
SORTED=s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14
RAW=s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56

# List available recordings (and matched raw files)
s3-to-phy list --sorted-uri $SORTED --raw-uri $RAW

# Convert all recordings to phy (streams from S3, compresses to .cbin)
s3-to-phy convert --sorted-uri $SORTED --raw-uri $RAW

# Convert only recordings for a specific probe
s3-to-phy convert --sorted-uri $SORTED --raw-uri $RAW --probe 46101

# Convert only a specific experiment
s3-to-phy convert --sorted-uri $SORTED --raw-uri $RAW --experiment experiment1

# Extract preprocessed traces as flat binary (for non-phy pipelines, e.g. mipmap)
s3-to-phy extract-recording \
  --analyzer-uri $SORTED/postprocessed/experiment1_...46101_recording1.zarr \
  --raw-uri $RAW/ecephys/ecephys_compressed/experiment1_...46101.zarr \
  --output preprocessed.dat

# Compress an existing recording.dat to .cbin
s3-to-phy compress --phy-dir ./phy_output
```

### Python API

```python
from aind_ecephys_s3_to_phy.analyzer import load_analyzer, attach_recording, discover_pairs
from aind_ecephys_s3_to_phy.recording import reconstruct_recording
from aind_ecephys_s3_to_phy.export import export_phy

# Discover and convert
pairs = discover_pairs(sorted_uri, raw_uri)
analyzer_name, raw_name = pairs[0]

analyzer_zarr = f"{sorted_uri}/postprocessed/{analyzer_name}"
raw_zarr = f"{raw_uri}/ecephys/ecephys_compressed/{raw_name}"

analyzer = load_analyzer(analyzer_zarr)
recording = reconstruct_recording(analyzer_zarr, raw_zarr)
attach_recording(analyzer, recording)
export_phy(analyzer, recording, "./phy_output", compress=True)

# Or just get the preprocessed recording for other pipelines
recording = reconstruct_recording(analyzer_zarr, raw_zarr)
# → SpikeInterface BaseRecording (filtered, bad channels removed, 373 ch)
```

### S3 credentials

Public buckets (like `aind-open-data`) work without credentials. For private buckets, configure AWS credentials via the standard methods (`~/.aws/credentials`, environment variables, IAM roles). SpikeInterface uses `s3fs`/`fsspec` which follows the standard AWS credential chain.

## Development

```bash
uv sync                                # Set up environment
./scripts/run_linters_and_checks.sh -c  # Full lint + test suite
uv run --frozen pytest -m integration   # Integration tests (requires S3)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

"""Shared fixtures and configuration for tests."""

import pytest

SORTED_ASSET = "s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14"
RAW_ASSET = "s3://aind-open-data/ecephys_786867_2025-09-25_12-43-56"
# A recording pair known to exist in the test dataset
TEST_ANALYZER = "experiment1_Record Node 111#Neuropix-PXI-100.50213-1_recording1.zarr"
TEST_RAW = "experiment1_Record Node 111#Neuropix-PXI-100.50213-1.zarr"


@pytest.fixture()
def sorted_asset_uri():
    return SORTED_ASSET


@pytest.fixture()
def raw_asset_uri():
    return RAW_ASSET


@pytest.fixture()
def analyzer_uri():
    return f"{SORTED_ASSET}/postprocessed/{TEST_ANALYZER}"


@pytest.fixture()
def raw_uri():
    return f"{RAW_ASSET}/ecephys/ecephys_compressed/{TEST_RAW}"

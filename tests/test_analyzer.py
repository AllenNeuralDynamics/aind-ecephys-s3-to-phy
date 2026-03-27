"""Unit tests for analyzer discovery helpers (no S3 access needed)."""

import pytest

from aind_ecephys_s3_to_phy.analyzer import filter_recordings
from aind_ecephys_s3_to_phy.recording import match_analyzer_to_raw

SAMPLE_NAMES = [
    "experiment1_Record Node 109#Neuropix-PXI-100.46101_recording1.zarr",
    "experiment1_Record Node 109#Neuropix-PXI-100.50198_recording1.zarr",
    "experiment1_Record Node 111#Neuropix-PXI-100.50213-1_recording1.zarr",
    "experiment2_Record Node 109#Neuropix-PXI-100.46101_recording1.zarr",
]


class TestMatchAnalyzerToRaw:
    @pytest.mark.parametrize(
        ("analyzer_name", "expected_raw"),
        [
            (
                "experiment1_Record Node 109#Neuropix-PXI-100.46101_recording1.zarr",
                "experiment1_Record Node 109#Neuropix-PXI-100.46101.zarr",
            ),
            (
                "experiment1_Record Node 111#Neuropix-PXI-100.50213-2_recording12.zarr",
                "experiment1_Record Node 111#Neuropix-PXI-100.50213-2.zarr",
            ),
            (
                "experiment1_Record Node 109#Neuropix-PXI-100.46101.zarr",
                "experiment1_Record Node 109#Neuropix-PXI-100.46101.zarr",
            ),
        ],
    )
    def test_match_names(self, analyzer_name, expected_raw):
        assert match_analyzer_to_raw(analyzer_name) == expected_raw


class TestFilterRecordings:
    def test_no_filters_returns_all(self):
        assert filter_recordings(SAMPLE_NAMES) == SAMPLE_NAMES

    def test_filter_by_probe(self):
        result = filter_recordings(SAMPLE_NAMES, probe="46101")
        assert len(result) == 2
        assert all("46101" in n for n in result)

    def test_filter_by_experiment(self):
        result = filter_recordings(SAMPLE_NAMES, experiment="experiment1")
        assert len(result) == 3
        assert all(n.startswith("experiment1") for n in result)

    def test_filter_by_both(self):
        result = filter_recordings(SAMPLE_NAMES, probe="46101", experiment="experiment2")
        assert result == ["experiment2_Record Node 109#Neuropix-PXI-100.46101_recording1.zarr"]

    def test_filter_no_match(self):
        result = filter_recordings(SAMPLE_NAMES, probe="99999")
        assert result == []

    def test_preserves_order(self):
        result = filter_recordings(SAMPLE_NAMES, experiment="experiment1")
        assert result == [n for n in SAMPLE_NAMES if n.startswith("experiment1")]

"""Unit tests for recording dict manipulation (no S3 access needed)."""

import copy

import pytest

from aind_ecephys_s3_to_phy.recording import fix_recording_dict, match_analyzer_to_raw

# A minimal nested recording dict mimicking the structure from a SortingAnalyzer.
SAMPLE_REC_DICT = {
    "class": "spikeinterface.core.channelslice.ChannelSliceRecording",
    "kwargs": {
        "channel_ids": ["CH0", "CH1"],
        "parent_recording": {
            "class": "spikeinterface.preprocessing.filter.HighpassFilterRecording",
            "kwargs": {
                "freq_min": 300.0,
                "recording": {
                    "class": "spikeinterface.core.zarrextractors.ZarrRecordingExtractor",
                    "kwargs": {
                        "folder_path": "../../../../data/raw.zarr",
                        "storage_options": None,
                    },
                    "relative_paths": True,
                    "version": "0.102.0",
                    "annotations": {},
                    "properties": {},
                },
            },
            "relative_paths": True,
            "version": "0.102.0",
            "annotations": {},
            "properties": {},
        },
    },
    "relative_paths": True,
    "version": "0.102.0",
    "annotations": {},
    "properties": {},
}


class TestFixRecordingDict:
    """Tests for fix_recording_dict."""

    def test_substitutes_leaf_path(self):
        result = fix_recording_dict(SAMPLE_REC_DICT, "s3://bucket/raw.zarr")
        leaf = result["kwargs"]["parent_recording"]["kwargs"]["recording"]
        assert leaf["kwargs"]["folder_path"] == "s3://bucket/raw.zarr"

    def test_clears_relative_paths(self):
        result = fix_recording_dict(SAMPLE_REC_DICT, "/local/raw.zarr")
        assert result["relative_paths"] is False
        assert result["kwargs"]["parent_recording"]["relative_paths"] is False
        leaf = result["kwargs"]["parent_recording"]["kwargs"]["recording"]
        assert leaf["relative_paths"] is False

    def test_preserves_intermediate_steps(self):
        result = fix_recording_dict(SAMPLE_REC_DICT, "s3://bucket/raw.zarr")
        # Outer ChannelSlice preserved
        assert "ChannelSlice" in result["class"]
        assert result["kwargs"]["channel_ids"] == ["CH0", "CH1"]
        # Middle HighpassFilter preserved
        middle = result["kwargs"]["parent_recording"]
        assert "HighpassFilter" in middle["class"]
        assert middle["kwargs"]["freq_min"] == 300.0

    def test_does_not_mutate_original(self):
        original = copy.deepcopy(SAMPLE_REC_DICT)
        fix_recording_dict(SAMPLE_REC_DICT, "s3://bucket/raw.zarr")
        assert SAMPLE_REC_DICT == original

    def test_deeply_nested_chain(self):
        """Test a 6-level chain matching real AIND preprocessing."""
        deep_dict = _make_dict(
            "ChannelSlice",
            "parent_recording",
            _make_dict(
                "CommonReference",
                "recording",
                _make_dict(
                    "ChannelSlice",
                    "parent_recording",
                    _make_dict(
                        "HighpassFilter",
                        "recording",
                        _make_dict(
                            "PhaseShift",
                            "recording",
                            _make_dict(
                                "SelectSegment",
                                "recording",
                                _make_leaf("old/path.zarr"),
                            ),
                        ),
                    ),
                ),
            ),
        )
        result = fix_recording_dict(deep_dict, "s3://new/path.zarr")
        # Walk to the leaf
        node = result
        for key in ("parent_recording", "recording", "parent_recording", "recording", "recording", "recording"):
            node = node["kwargs"][key]
        assert node["kwargs"]["folder_path"] == "s3://new/path.zarr"

    def test_handles_parent_recording_key(self):
        """ChannelSlice uses 'parent_recording', filter uses 'recording'."""
        dict_with_parent = _make_dict("ChannelSlice", "parent_recording", _make_leaf("old.zarr"))
        result = fix_recording_dict(dict_with_parent, "new.zarr")
        assert result["kwargs"]["parent_recording"]["kwargs"]["folder_path"] == "new.zarr"

        dict_with_recording = _make_dict("Filter", "recording", _make_leaf("old.zarr"))
        result = fix_recording_dict(dict_with_recording, "new.zarr")
        assert result["kwargs"]["recording"]["kwargs"]["folder_path"] == "new.zarr"


def _make_leaf(path: str) -> dict:
    return {
        "class": "spikeinterface.core.zarrextractors.ZarrRecordingExtractor",
        "kwargs": {"folder_path": path, "storage_options": None},
        "relative_paths": True,
        "version": "0.102.0",
        "annotations": {},
        "properties": {},
    }


def _make_dict(class_name: str, child_key: str, child: dict) -> dict:
    return {
        "class": f"spikeinterface.preprocessing.{class_name}",
        "kwargs": {child_key: child},
        "relative_paths": True,
        "version": "0.102.0",
        "annotations": {},
        "properties": {},
    }


class TestMatchAnalyzerToRaw:
    """Tests for match_analyzer_to_raw."""

    @pytest.mark.parametrize(
        ("analyzer_name", "expected"),
        [
            ("experiment1_Neuropix-PXI-100.50213-1_recording1.zarr", "experiment1_Neuropix-PXI-100.50213-1.zarr"),
            ("experiment1_Neuropix-PXI-100.50213-1_recording12.zarr", "experiment1_Neuropix-PXI-100.50213-1.zarr"),
            ("experiment1_Neuropix-PXI-100.50213-1.zarr", "experiment1_Neuropix-PXI-100.50213-1.zarr"),
        ],
    )
    def test_strips_recording_suffix(self, analyzer_name, expected):
        assert match_analyzer_to_raw(analyzer_name) == expected

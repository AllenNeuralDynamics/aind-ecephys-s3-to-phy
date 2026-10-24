"""Tests for the CLI's motion-correction step."""

import numpy as np
import pytest
import spikeinterface as si
from spikeinterface.preprocessing import astype

from aind_ecephys_s3_to_phy.cli import _maybe_motion_correct

REC = "experiment1_Record Node 104#Neuropix-PXI-100.50213-1_recording1"
T0, DURATION_S = 5000.0, 20.0


@pytest.fixture()
def recording():
    rec = si.generate_recording(num_channels=16, durations=[DURATION_S], seed=0)
    rec.shift_times(T0)
    return astype(rec, "int16")


@pytest.fixture()
def sorted_asset(tmp_path):
    (tmp_path / "postprocessed").mkdir()
    folder = tmp_path / "spikesorted/motion" / REC
    folder.mkdir(parents=True)
    bins = T0 + np.arange(1, DURATION_S, 2.0)
    np.save(folder / "displacement_seg0.npy", np.full((bins.size, 1), 10.0))
    np.save(folder / "temporal_bins_s_seg0.npy", bins)
    np.save(folder / "spatial_bins_um.npy", np.zeros(1))
    return tmp_path


def test_ks4_correction_keeps_int16_and_channels(recording, sorted_asset):
    corrected = _maybe_motion_correct(recording, str(sorted_asset), f"{REC}.zarr", motion_correct=True)
    assert corrected is not recording
    assert corrected.get_dtype() == np.int16
    assert list(corrected.channel_ids) == list(recording.channel_ids)
    assert not np.array_equal(corrected.get_traces(end_frame=1000), recording.get_traces(end_frame=1000))


def test_missing_ks4_estimate_leaves_traces(recording, tmp_path):
    (tmp_path / "postprocessed").mkdir()
    assert _maybe_motion_correct(recording, str(tmp_path), f"{REC}.zarr", motion_correct=True) is recording


def test_disabled(recording, sorted_asset):
    assert _maybe_motion_correct(recording, str(sorted_asset), REC, motion_correct=False) is recording

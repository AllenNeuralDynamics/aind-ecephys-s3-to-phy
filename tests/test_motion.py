"""Tests for motion correction discovery, loading, and application."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core.generate import generate_recording
from spikeinterface.core.motion import Motion

from aind_ecephys_s3_to_phy.motion import apply_motion_correction, find_motion_path, load_motion

# --- Helpers ---


def _write_motion(motion_dir, *, n_time_bins=10, n_spatial_bins=3, duration_s=60.0):
    """Write a synthetic Motion to *motion_dir* and return the Motion object."""
    temporal_bins = np.linspace(0, duration_s, n_time_bins)
    spatial_bins = np.linspace(0, 3000, n_spatial_bins)
    # Sinusoidal displacement: ~20 um peak
    displacement = 20.0 * np.sin(2 * np.pi * temporal_bins[:, None] / duration_s) * np.ones(n_spatial_bins)

    motion = Motion(
        displacement=displacement,
        temporal_bins_s=temporal_bins,
        spatial_bins_um=spatial_bins,
    )
    motion.save(motion_dir)
    return motion


# --- find_motion_path ---


class TestFindMotionPath:
    def test_found(self, tmp_path):
        """Returns path when motion estimate exists."""
        rec_name = "experiment1_Record Node 109#Neuropix-PXI-100.50720_recording1.zarr"
        rec_stem = rec_name.removesuffix(".zarr")
        motion_dir = tmp_path / "preprocessed" / "motion" / rec_stem / "motion"
        _write_motion(motion_dir)

        result = find_motion_path(str(tmp_path), rec_name)
        assert result is not None
        assert result.endswith("/motion")

    def test_not_found(self, tmp_path):
        """Returns None when no motion estimate exists."""
        result = find_motion_path(str(tmp_path), "nonexistent_recording1.zarr")
        assert result is None

    def test_empty_dir(self, tmp_path):
        """Returns None when the directory exists but has no marker file."""
        rec_name = "experiment1_rec_recording1.zarr"
        rec_stem = rec_name.removesuffix(".zarr")
        motion_dir = tmp_path / "preprocessed" / "motion" / rec_stem / "motion"
        motion_dir.mkdir(parents=True)
        # No spikeinterface_info.json
        result = find_motion_path(str(tmp_path), rec_name)
        assert result is None


# --- load_motion ---


class TestLoadMotion:
    def test_roundtrip(self, tmp_path):
        """load_motion returns a Motion object matching what was saved."""
        motion_dir = tmp_path / "motion"
        original = _write_motion(motion_dir)
        loaded = load_motion(str(motion_dir))
        assert loaded == original

    def test_bad_path(self, tmp_path):
        """Raises when the path doesn't contain a valid Motion."""
        with pytest.raises((IOError, OSError)):
            load_motion(str(tmp_path / "nonexistent"))


# --- apply_motion_correction ---


def _make_recording_with_probe(num_channels=32, duration_s=10.0):
    """Create a synthetic recording with a linear probe."""
    from probeinterface import generate_linear_probe

    recording = generate_recording(
        num_channels=num_channels,
        durations=[duration_s],
        sampling_frequency=30000.0,
        set_probe=False,
    )
    probe = generate_linear_probe(num_elec=num_channels, ypitch=20.0)
    probe.set_device_channel_indices(np.arange(num_channels))
    return recording.set_probe(probe)


class TestApplyMotionCorrection:
    def test_returns_recording_same_channels(self):
        """Output recording has the same number of channels as input (force_extrapolate)."""
        recording = _make_recording_with_probe(32, 10.0)

        motion = Motion(
            displacement=np.ones((5, 1)) * 10.0,  # rigid 10 um shift
            temporal_bins_s=np.linspace(0, 10, 5),
            spatial_bins_um=np.array([320.0]),  # single bin = rigid
        )

        corrected = apply_motion_correction(recording, motion)
        assert corrected.get_num_channels() == recording.get_num_channels()
        assert corrected.get_num_frames() == recording.get_num_frames()

    def test_traces_differ(self):
        """Motion correction actually changes the traces."""
        recording = _make_recording_with_probe(32, 2.0)

        # Large displacement so traces are visibly different
        motion = Motion(
            displacement=np.ones((5, 1)) * 40.0,
            temporal_bins_s=np.linspace(0, 2, 5),
            spatial_bins_um=np.array([320.0]),
        )

        corrected = apply_motion_correction(recording, motion)
        original_traces = recording.get_traces(start_frame=0, end_frame=1000)
        corrected_traces = corrected.get_traces(start_frame=0, end_frame=1000)

        # With a 40 um shift (2 electrode pitches), the traces should differ
        assert not np.array_equal(original_traces, corrected_traces)

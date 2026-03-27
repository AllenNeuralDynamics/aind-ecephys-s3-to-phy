"""Tests for the export pipeline."""

import numpy as np
import pytest

from aind_ecephys_s3_to_phy.export import _read_params, _update_params_dat_path, compress_binary

SAMPLE_PARAMS = """\
dat_path = r'recording.dat'
n_channels_dat = 10
dtype = 'int16'
offset = 0
sample_rate = 30000.0
hp_filtered = True
"""

SAMPLE_PARAMS_NONE = """\
dat_path = None
n_channels_dat = 10
dtype = 'int16'
offset = 0
sample_rate = 30000.0
hp_filtered = True
"""


class TestReadParams:
    def test_reads_valid_params(self, tmp_path):
        (tmp_path / "params.py").write_text(SAMPLE_PARAMS)
        params = _read_params(tmp_path)
        assert params["n_channels_dat"] == 10
        assert params["dtype"] == "int16"
        assert params["sample_rate"] == 30000.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _read_params(tmp_path)


class TestUpdateParamsDatPath:
    def test_replaces_dat_path(self, tmp_path):
        (tmp_path / "params.py").write_text(SAMPLE_PARAMS)
        _update_params_dat_path(tmp_path, "recording.cbin")
        text = (tmp_path / "params.py").read_text()
        assert "dat_path = r'recording.cbin'" in text
        # Other lines preserved
        assert "n_channels_dat = 10" in text
        assert "sample_rate = 30000.0" in text

    def test_replaces_none_dat_path(self, tmp_path):
        (tmp_path / "params.py").write_text(SAMPLE_PARAMS_NONE)
        _update_params_dat_path(tmp_path, "recording.dat")
        text = (tmp_path / "params.py").read_text()
        assert "dat_path = r'recording.dat'" in text


class TestCompressBinary:
    def test_missing_dat_raises(self, tmp_path):
        (tmp_path / "params.py").write_text(SAMPLE_PARAMS)
        with pytest.raises(FileNotFoundError, match="recording.dat"):
            compress_binary(tmp_path)

    def test_compress_synthetic_dat(self, tmp_path):
        n_channels = 10
        n_frames = 100
        dtype = np.dtype("int16")

        # Write synthetic recording.dat
        data = np.random.default_rng(42).integers(-1000, 1000, size=(n_frames, n_channels), dtype=dtype)
        dat_path = tmp_path / "recording.dat"
        data.tofile(dat_path)

        # Write matching params.py
        (tmp_path / "params.py").write_text(SAMPLE_PARAMS)

        compress_binary(tmp_path)

        # .dat deleted, .cbin and .ch created
        assert not dat_path.exists()
        assert (tmp_path / "recording.cbin").exists()
        assert (tmp_path / "recording.ch").exists()

        # params.py updated
        text = (tmp_path / "params.py").read_text()
        assert "recording.cbin" in text

        # .cbin is readable
        import mtscomp

        reader = mtscomp.Reader()
        reader.open(str(tmp_path / "recording.cbin"), str(tmp_path / "recording.ch"))
        readback = reader[:]
        assert readback.shape == (n_frames, n_channels)
        np.testing.assert_array_equal(readback, data)
        reader.close()


# --- Integration tests (require S3) ---


@pytest.mark.integration
def test_single_recording_spike_export(analyzer_uri, raw_uri, tmp_path):
    """Load analyzer from S3, reconstruct recording, export spike data only."""
    from spikeinterface.exporters import export_to_phy

    from aind_ecephys_s3_to_phy.analyzer import attach_recording, load_analyzer
    from aind_ecephys_s3_to_phy.recording import reconstruct_recording

    analyzer = load_analyzer(analyzer_uri)
    recording = reconstruct_recording(analyzer_uri, raw_uri)

    assert recording.get_num_channels() > 0
    assert recording.is_filtered()

    attach_recording(analyzer, recording)
    assert analyzer.has_recording()

    output_dir = tmp_path / "phy_output"
    export_to_phy(
        analyzer,
        output_dir,
        copy_binary=False,
        compute_pc_features=False,
        compute_amplitudes=False,
    )

    assert (output_dir / "spike_times.npy").exists()
    assert (output_dir / "spike_clusters.npy").exists()
    assert (output_dir / "templates.npy").exists()
    assert (output_dir / "params.py").exists()
    assert (output_dir / "channel_positions.npy").exists()

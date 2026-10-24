"""Tests for the CLI entry point."""

import subprocess
import sys

import pytest


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "aind_ecephys_s3_to_phy.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "convert" in result.stdout
    assert "list" in result.stdout
    assert "extract-recording" in result.stdout
    assert "compress" in result.stdout


def test_convert_help():
    result = subprocess.run(
        [sys.executable, "-m", "aind_ecephys_s3_to_phy.cli", "convert", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--sorted-uri" in result.stdout
    assert "--raw-uri" in result.stdout
    assert "--probe" in result.stdout
    assert "--experiment" in result.stdout
    assert "--lazy" in result.stdout


def test_extract_recording_help():
    result = subprocess.run(
        [sys.executable, "-m", "aind_ecephys_s3_to_phy.cli", "extract-recording", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--analyzer-uri" in result.stdout
    assert "--raw-uri" in result.stdout
    assert "--output" in result.stdout


def test_compress_help():
    result = subprocess.run(
        [sys.executable, "-m", "aind_ecephys_s3_to_phy.cli", "compress", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--phy-dir" in result.stdout


class TestCleanupPartial:
    def test_removes_dat(self, tmp_path):
        from aind_ecephys_s3_to_phy.cli import _cleanup_partial

        (tmp_path / "recording.dat").write_bytes(b"\x00" * 100)
        (tmp_path / "params.py").write_text("dat_path = r'recording.dat'\n")
        _cleanup_partial(tmp_path)
        assert not (tmp_path / "recording.dat").exists()
        # params.py should still be there
        assert (tmp_path / "params.py").exists()

    def test_no_dat_no_crash(self, tmp_path):
        from aind_ecephys_s3_to_phy.cli import _cleanup_partial

        (tmp_path / "params.py").write_text("dat_path = None\n")
        _cleanup_partial(tmp_path)  # should not raise

    def test_nonexistent_dir_no_crash(self, tmp_path):
        from aind_ecephys_s3_to_phy.cli import _cleanup_partial

        _cleanup_partial(tmp_path / "does_not_exist")  # should not raise


def test_convert_help_includes_motion_correct():
    result = subprocess.run(
        [sys.executable, "-m", "aind_ecephys_s3_to_phy.cli", "convert", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--motion-correct" in result.stdout or "--no-motion-correct" in result.stdout


def test_extract_recording_help_includes_motion_correct():
    result = subprocess.run(
        [sys.executable, "-m", "aind_ecephys_s3_to_phy.cli", "extract-recording", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--motion-correct" in result.stdout or "--no-motion-correct" in result.stdout


@pytest.mark.integration
def test_list_subcommand(sorted_asset_uri):
    result = subprocess.run(
        [sys.executable, "-m", "aind_ecephys_s3_to_phy.cli", "list", "--sorted-uri", sorted_asset_uri],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0
    assert "recording1.zarr" in result.stdout

"""Example test module."""

import aind_ecephys_s3_to_phy


def test_version():
    """Test that version is defined."""
    assert aind_ecephys_s3_to_phy.__version__ is not None
    assert isinstance(aind_ecephys_s3_to_phy.__version__, str)

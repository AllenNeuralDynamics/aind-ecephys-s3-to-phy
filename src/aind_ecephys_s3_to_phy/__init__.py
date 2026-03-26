"""Tools to get phy-ready data from AIND ecephys data on S3"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aind-ecephys-s3-to-phy")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

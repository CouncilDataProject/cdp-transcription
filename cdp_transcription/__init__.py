"""Top-level package for cdp_transcription."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cdp-transcription")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"

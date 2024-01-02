"""
Re-export the version from the core module written in Rust.
"""
from ._core import __version__

__all__ = ("VERSION",)

VERSION = __version__
"""The current version of this module."""

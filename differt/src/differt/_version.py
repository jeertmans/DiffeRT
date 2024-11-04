"""Re-export the version from the core module written in Rust."""

from differt_core import __version__

__all__ = ("VERSION",)

VERSION: str = __version__
"""The current version of this module."""

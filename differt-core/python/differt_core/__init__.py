"""
Core package written in Rust and re-exported here.

The present package only provides lower-level utilities with
a focus on performances.

Currently, Rust uses NumPy's C API to communicate between Python
and Rust, hence casting NumPy arrays to JAX arrays in the process,
thus making it impossible to trace variables used in Rust code.

In the future, we plan on providing XLA-compatible functions,
so that one can use :mod:`differt_core` and still be able to differentiate
its code. We welcome any contribution on that topic!
"""

from differt_core._differt_core import __version__ as _version
from differt_core._differt_core import __version_info__ as _version_info

__all__ = ("__version__", "__version_info__")

__version__ = _version
"""The current full version of this module."""
__version_info__ = _version_info
"""The current short version of this module as a tuple (major, minor, patch)."""

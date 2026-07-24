__all__ = ("Mesh", "TriangleMesh")


import warnings
from typing import Any, Self

from differt_core import _differt_core

Mesh = _differt_core.geometry.mesh.Mesh


# Deprecated alias
class TriangleMesh(Mesh):
    """
    Deprecated alias for :class:`Mesh`.

    .. deprecated:: 0.10
        Use :class:`Mesh` instead.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Deprecated constructor."""
        warnings.warn(
            "TriangleMesh is deprecated, use Mesh instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls, *args, **kwargs)

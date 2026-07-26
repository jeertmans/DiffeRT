__all__ = ("Scene", "TriangleScene")

import warnings
from typing import Any, Self

from differt_core import _differt_core

Scene = _differt_core.geometry.scene.Scene


# Deprecated alias
class TriangleScene(Scene):
    """
    Deprecated alias for :class:`Scene`.

    .. deprecated:: 0.10
        Use :class:`Scene` instead.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Deprecated constructor."""
        warnings.warn(
            "TriangleScene is deprecated, use Scene instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls, *args, **kwargs)

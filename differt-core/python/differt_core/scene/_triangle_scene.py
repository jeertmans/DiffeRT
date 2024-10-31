"""Triangle scene utilities used by :mod:`differt.scene`."""

__all__ = ("TriangleScene",)

from differt_core import _lowlevel

TriangleScene = _lowlevel.scene.triangle_scene.TriangleScene

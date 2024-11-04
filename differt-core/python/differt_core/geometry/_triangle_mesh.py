"""Triangle mesh utilities used by :mod:`differt.geometry.triangle_mesh`."""

__all__ = ("TriangleMesh",)

from differt_core import _lowlevel

TriangleMesh = _lowlevel.geometry.triangle_mesh.TriangleMesh

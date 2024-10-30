"""Geometries for building scenes."""

__all__ = ("TriangleMesh", "Paths", "triangles_contain_vertices_assuming_inside_same_plane", "fibonacci_lattice")

from ._triangle_mesh import TriangleMesh, triangles_contain_vertices_assuming_inside_same_plane
from ._utils import fibonacci_lattice
from ._paths import Paths

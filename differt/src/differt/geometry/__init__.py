"""Geometries for building scenes."""

__all__ = (
    "Paths",
    "TriangleMesh",
    "assemble_paths",
    "fibonacci_lattice",
    "normalize",
    "orthogonal_basis",
    "path_lengths",
    "rotation_matrix_along_axis",
    "rotation_matrix_along_x_axis",
    "rotation_matrix_along_x_axis",
    "rotation_matrix_along_y_axis",
    "rotation_matrix_along_z_axis",
    "triangles_contain_vertices_assuming_inside_same_plane",
)

from ._paths import Paths
from ._triangle_mesh import (
    TriangleMesh,
    triangles_contain_vertices_assuming_inside_same_plane,
)
from ._utils import (
    assemble_paths,
    fibonacci_lattice,
    normalize,
    orthogonal_basis,
    path_lengths,
    rotation_matrix_along_axis,
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
)

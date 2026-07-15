"""Geometries for building scenes."""

__all__ = (
    "LaunchedPaths",
    "Mesh",
    "Paths",
    "SBRPaths",
    "TracedPaths",
    "TriangleMesh",
    "assemble_path",
    "cartesian_to_spherical",
    "fibonacci_lattice",
    "merge_cell_ids",
    "min_distance_between_cells",
    "normalize",
    "orthogonal_basis",
    "path_length",
    "perpendicular_vector",
    "rotation_matrix_along_axis",
    "rotation_matrix_along_x_axis",
    "rotation_matrix_along_y_axis",
    "rotation_matrix_along_z_axis",
    "spherical_to_cartesian",
    "triangle_contains_vertex_assuming_inside_same_plane",
    "viewing_frustum",
)

from ._mesh import (
    Mesh,
    TriangleMesh,
    triangle_contains_vertex_assuming_inside_same_plane,
)
from ._paths import LaunchedPaths, Paths, SBRPaths, TracedPaths, merge_cell_ids
from ._utils import (
    assemble_path,
    cartesian_to_spherical,
    fibonacci_lattice,
    min_distance_between_cells,
    normalize,
    orthogonal_basis,
    path_length,
    perpendicular_vector,
    rotation_matrix_along_axis,
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
    spherical_to_cartesian,
    viewing_frustum,
)

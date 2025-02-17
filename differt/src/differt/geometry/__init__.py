"""Geometries for building scenes."""

__all__ = (
    "Paths",
    "SBRPaths",
    "TriangleMesh",
    "assemble_paths",
    "cartesian_to_spherical",
    "fibonacci_lattice",
    "merge_cell_ids",
    "min_distance_between_cells",
    "normalize",
    "orthogonal_basis",
    "pairwise_cross",
    "path_lengths",
    "perpendicular_vectors",
    "rotation_matrix_along_axis",
    "rotation_matrix_along_x_axis",
    "rotation_matrix_along_x_axis",
    "rotation_matrix_along_y_axis",
    "rotation_matrix_along_z_axis",
    "spherical_to_cartesian",
    "triangles_contain_vertices_assuming_inside_same_plane",
    "viewing_frustum",
)

from ._paths import Paths, SBRPaths, merge_cell_ids
from ._triangle_mesh import (
    TriangleMesh,
    triangles_contain_vertices_assuming_inside_same_plane,
)
from ._utils import (
    assemble_paths,
    cartesian_to_spherical,
    fibonacci_lattice,
    min_distance_between_cells,
    normalize,
    orthogonal_basis,
    pairwise_cross,
    path_lengths,
    perpendicular_vectors,
    rotation_matrix_along_axis,
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
    spherical_to_cartesian,
    viewing_frustum,
)

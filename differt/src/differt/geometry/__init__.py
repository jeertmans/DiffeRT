"""Geometries for building scenes."""

__all__ = (
    "LaunchPaths",
    "Paths",
    "SBRPaths",
    "TracePaths",
    "TriangleMesh",
    "assemble_path",
    "cantor_pair",
    "cartesian_to_spherical",
    "fibonacci_lattice",
    "fuse_ray_bundles",
    "hash_interaction_sequence",
    "interception_plane_check",
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

from ._hashing import cantor_pair, hash_interaction_sequence
from ._interception import fuse_ray_bundles, interception_plane_check
from ._paths import LaunchPaths, Paths, SBRPaths, TracePaths, merge_cell_ids
from ._triangle_mesh import (
    TriangleMesh,
    triangle_contains_vertex_assuming_inside_same_plane,
)
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


"""Geometries for building scenes."""

__all__ = (
    "ExhaustivePathSolver",
    "HybridPathSolver",
    "LaunchPaths",
    "PathSolverConfig",
    "Paths",
    "SBRPaths",
    "SBRPathSolver",
    "SizedIterator",
    "TracePaths",
    "TriangleMesh",
    "TriangleScene",
    "assemble_path",
    "cartesian_to_spherical",
    "consecutive_vertices_are_on_same_side_of_mirror",
    "download_sionna_scenes",
    "fermat_path_on_linear_objects",
    "fermat_path_on_planar_mirrors",
    "fibonacci_lattice",
    "first_triangle_hit_by_ray",
    "generate_all_path_candidates",
    "generate_all_path_candidates_chunks_iter",
    "generate_all_path_candidates_iter",
    "get_sionna_scene",
    "image_method",
    "image_of_vertex_with_respect_to_mirror",
    "intersection_of_ray_with_plane",
    "list_sionna_scenes",
    "merge_cell_ids",
    "min_distance_between_cells",
    "normalize",
    "orthogonal_basis",
    "path_length",
    "perpendicular_vector",
    "ray_intersect_any_triangle",
    "ray_intersect_triangle",
    "rotation_matrix_along_axis",
    "rotation_matrix_along_x_axis",
    "rotation_matrix_along_y_axis",
    "rotation_matrix_along_z_axis",
    "spherical_to_cartesian",
    "triangle_contains_vertex_assuming_inside_same_plane",
    "triangles_visible_from_vertex",
    "viewing_frustum",
)

from ._fermat import fermat_path_on_linear_objects, fermat_path_on_planar_mirrors
from ._image_method import (
    consecutive_vertices_are_on_same_side_of_mirror,
    image_method,
    image_of_vertex_with_respect_to_mirror,
    intersection_of_ray_with_plane,
)
from ._paths import LaunchPaths, Paths, SBRPaths, TracePaths, merge_cell_ids
from ._rt_utils import (
    SizedIterator,
    first_triangle_hit_by_ray,
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    generate_all_path_candidates_iter,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
    triangles_visible_from_vertex,
)
from ._sionna import download_sionna_scenes, get_sionna_scene, list_sionna_scenes
from ._solvers import (
    ExhaustivePathSolver,
    HybridPathSolver,
    PathSolverConfig,
    SBRPathSolver,
)
from ._triangle_mesh import (
    TriangleMesh,
    triangle_contains_vertex_assuming_inside_same_plane,
)
from ._triangle_scene import TriangleScene
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


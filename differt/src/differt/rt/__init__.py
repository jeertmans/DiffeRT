"""Deprecated module."""

# ruff:file-ignore[non-empty-init-module, module-import-not-at-top-of-file]
import warnings

warnings.warn(
    "The differt.rt module is deprecated and will be removed in a future version. "
    "Please use differt.geometry instead.",
    DeprecationWarning,
    stacklevel=2,
)

from differt.geometry import (
    SizedIterator,
    consecutive_vertices_are_on_same_side_of_mirror,
    fermat_path_on_linear_objects,
    fermat_path_on_planar_mirrors,
    first_triangle_hit_by_ray,
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    generate_all_path_candidates_iter,
    image_method,
    image_of_vertex_with_respect_to_mirror,
    intersection_of_ray_with_plane,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
    triangles_visible_from_vertex,
)

__all__ = (
    "SizedIterator",
    "consecutive_vertices_are_on_same_side_of_mirror",
    "fermat_path_on_linear_objects",
    "fermat_path_on_planar_mirrors",
    "first_triangle_hit_by_ray",
    "generate_all_path_candidates",
    "generate_all_path_candidates_chunks_iter",
    "generate_all_path_candidates_iter",
    "image_method",
    "image_of_vertex_with_respect_to_mirror",
    "intersection_of_ray_with_plane",
    "ray_intersect_any_triangle",
    "ray_intersect_triangle",
    "triangles_visible_from_vertex",
)

"""Ray Tracing utilities."""

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

from ._fermat import fermat_path_on_linear_objects, fermat_path_on_planar_mirrors
from ._image_method import (
    consecutive_vertices_are_on_same_side_of_mirror,
    image_method,
    image_of_vertex_with_respect_to_mirror,
    intersection_of_ray_with_plane,
)
from ._utils import (
    SizedIterator,
    first_triangle_hit_by_ray,
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    generate_all_path_candidates_iter,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
    triangles_visible_from_vertex,
)

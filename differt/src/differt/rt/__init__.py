"""Ray Tracing utilities."""

__all__ = (
    "SizedIterator",
    "consecutive_vertices_are_on_same_side_of_mirrors",
    "fermat_path_on_planar_mirrors",
    "first_triangles_hit_by_rays",
    "generate_all_path_candidates",
    "generate_all_path_candidates_chunks_iter",
    "generate_all_path_candidates_iter",
    "image_method",
    "image_of_vertices_with_respect_to_mirrors",
    "intersection_of_rays_with_planes",
    "rays_intersect_any_triangle",
    "rays_intersect_triangles",
    "triangles_visible_from_vertices",
)

from ._fermat import fermat_path_on_planar_mirrors
from ._image_method import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    image_method,
    image_of_vertices_with_respect_to_mirrors,
    intersection_of_rays_with_planes,
)
from ._utils import (
    SizedIterator,
    first_triangles_hit_by_rays,
    generate_all_path_candidates,
    generate_all_path_candidates_chunks_iter,
    generate_all_path_candidates_iter,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
    triangles_visible_from_vertices,
)

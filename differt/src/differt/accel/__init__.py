"""Acceleration structures for ray tracing.

This module provides BVH (Bounding Volume Hierarchy) acceleration for
DiffeRT's ray-triangle intersection queries.

Example:
    Build a BVH over two triangles forming a quad, then shoot rays at it:

    >>> import jax.numpy as jnp
    >>> from differt.accel import TriangleBvh
    >>> verts = jnp.array([
    ...     [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     [[1, 1, 0], [1, 0, 0], [0, 1, 0]],
    ... ], dtype=jnp.float32)
    >>> bvh = TriangleBvh(verts)
    >>> bvh.num_triangles
    2
    >>> origins = jnp.array([[0.25, 0.25, 1.0], [5.0, 5.0, 1.0]])
    >>> dirs = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    >>> hit_idx, hit_t = bvh.nearest_hit(origins, dirs)
    >>> hit_idx
    array([ 0, -1], dtype=int32)
    >>> hit_t
    array([ 1., inf], dtype=float32)
"""

__all__ = (
    "TriangleBvh",
    "bvh_first_triangles_hit_by_rays",
    "bvh_rays_intersect_any_triangle",
    "bvh_triangles_visible_from_vertices",
)

from differt.accel._accelerated import (
    bvh_first_triangles_hit_by_rays,
    bvh_rays_intersect_any_triangle,
    bvh_triangles_visible_from_vertices,
)
from differt.accel._bvh import TriangleBvh

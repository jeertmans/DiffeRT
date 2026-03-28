"""Acceleration structures for ray tracing.

This module provides BVH (Bounding Volume Hierarchy) acceleration for
DiffeRT's ray-triangle intersection queries.

Example:
    >>> import jax.numpy as jnp
    >>> from differt.accel import TriangleBvh
    >>> verts = jnp.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=jnp.float32)
    >>> bvh = TriangleBvh(verts)
    >>> bvh.num_triangles
    1
"""

__all__ = (
    "TriangleBvh",
    "bvh_first_triangles_hit_by_rays",
    "bvh_rays_intersect_any_triangle",
)

from differt.accel._accelerated import (
    bvh_first_triangles_hit_by_rays,
    bvh_rays_intersect_any_triangle,
)
from differt.accel._bvh import TriangleBvh

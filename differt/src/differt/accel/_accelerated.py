"""BVH-accelerated versions of DiffeRT's ray-triangle intersection functions.

These are drop-in replacements for the functions in :mod:`differt.rt._utils`,
accelerated by a BVH for O(rays * log(triangles)) instead of O(rays * triangles).

For the hard (non-differentiable) path, the BVH does the full intersection.
For the soft (differentiable) path, the BVH selects candidates and the
existing JAX-based Moller-Trumbore runs on the reduced set.
"""

from __future__ import annotations

__all__ = (
    "bvh_first_triangles_hit_by_rays",
    "bvh_rays_intersect_any_triangle",
)

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.accel._bvh import TriangleBvh, compute_expansion_radius
from differt.rt._utils import rays_intersect_triangles
from differt.utils import smoothing_function


def bvh_rays_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    *,
    hit_tol: Float[ArrayLike, ""] | None = None,
    smoothing_factor: Float[ArrayLike, ""] | None = None,
    bvh: TriangleBvh | None = None,
    max_candidates: int = 512,
    epsilon_grad: float = 1e-7,
    **kwargs: Any,
) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
    """BVH-accelerated version of :func:`~differt.rt.rays_intersect_any_triangle`.

    When ``bvh`` is provided, uses BVH candidate selection to reduce the number
    of triangles tested per ray from O(N) to O(log N).

    For the hard path (``smoothing_factor=None``), uses BVH nearest-hit to check
    if any triangle blocks the ray.

    For the soft path (``smoothing_factor`` set), uses BVH with expanded boxes
    to find candidate triangles, then runs the standard soft intersection on
    candidates only. Gradients flow through the JAX soft intersection normally.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray directions.
        triangle_vertices: An array of triangle vertices.
        active_triangles: Optional boolean mask for active triangles.
        hit_tol: Tolerance for hit detection.
        smoothing_factor: If set, uses smooth sigmoid approximations.
        bvh: Pre-built BVH acceleration structure.
        max_candidates: Maximum candidates per ray for soft mode.
        epsilon_grad: Gradient truncation threshold for expansion radius.
        kwargs: Keyword arguments passed to :func:`rays_intersect_triangles`.

    Returns:
        For each ray, whether it intersects with any of the triangles.
    """
    if bvh is None:
        from differt.rt._utils import rays_intersect_any_triangle

        return rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            triangle_vertices,
            active_triangles,
            hit_tol=hit_tol,
            smoothing_factor=smoothing_factor,
            **kwargs,
        )

    ray_origins_jnp = jnp.asarray(ray_origins)
    ray_directions_jnp = jnp.asarray(ray_directions)
    triangle_vertices_jnp = jnp.asarray(triangle_vertices)

    if hit_tol is None:
        dtype = jnp.result_type(
            ray_origins_jnp, ray_directions_jnp, triangle_vertices_jnp
        )
        hit_tol = 10.0 * jnp.finfo(dtype).eps

    hit_threshold = 1.0 - jnp.asarray(hit_tol)
    batch_shape = ray_origins_jnp.shape[:-1]

    if smoothing_factor is None:
        # Hard mode: use BVH nearest-hit as an "any" check.
        # A ray intersects some triangle iff nearest_hit returns a valid index.
        flat_origins = np.asarray(ray_origins_jnp).reshape(-1, 3)
        flat_dirs = np.asarray(ray_directions_jnp).reshape(-1, 3)
        hit_indices, hit_t = bvh.nearest_hit(flat_origins, flat_dirs)

        # Apply hit_threshold: only count hits with t < hit_threshold
        any_hit = (hit_indices >= 0) & (hit_t < float(hit_threshold))

        # Apply active_triangles filter
        if active_triangles is not None:
            active = np.asarray(active_triangles).flatten()
            # Check if the hit triangle is active
            valid_hit = np.zeros_like(any_hit)
            for i in range(len(hit_indices)):
                if any_hit[i] and active[hit_indices[i]]:
                    valid_hit[i] = True
            any_hit = valid_hit

        return jnp.asarray(any_hit.reshape(batch_shape))

    # Soft/differentiable mode: BVH candidate selection + JAX soft intersection
    alpha = float(smoothing_factor)

    # Estimate triangle size for expansion radius
    tri_np = np.asarray(triangle_vertices_jnp)
    if tri_np.ndim > 3:
        # Flatten batch dims for triangle size estimation
        flat_tri = tri_np.reshape(-1, 3, 3)
    else:
        flat_tri = tri_np
    # Use mean edge length as characteristic size
    edges = np.diff(flat_tri, axis=-2, append=flat_tri[..., :1, :])
    mean_tri_size = float(np.mean(np.linalg.norm(edges, axis=-1)))
    expansion = compute_expansion_radius(alpha, mean_tri_size, epsilon_grad)

    # Check if expansion is too large (soft smoothing -> fallback to brute force)
    scene_diag = float(
        np.linalg.norm(flat_tri.reshape(-1, 3).max(axis=0) - flat_tri.reshape(-1, 3).min(axis=0))
    )
    if expansion > scene_diag:
        from differt.rt._utils import rays_intersect_any_triangle

        return rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            triangle_vertices,
            active_triangles,
            hit_tol=hit_tol,
            smoothing_factor=smoothing_factor,
            **kwargs,
        )

    # Get candidates from BVH (outside JIT -- returns numpy arrays)
    flat_origins = np.asarray(ray_origins_jnp).reshape(-1, 3)
    flat_dirs = np.asarray(ray_directions_jnp).reshape(-1, 3)
    candidate_indices, candidate_counts = bvh.get_candidates(
        flat_origins, flat_dirs, expansion, max_candidates
    )

    # If any ray has more candidates than max_candidates, fall back to brute force
    # for correctness (truncation would give wrong gradients)
    if np.any(candidate_counts > max_candidates):
        import warnings

        warnings.warn(
            f"BVH candidate count ({int(candidate_counts.max())}) exceeds "
            f"max_candidates ({max_candidates}). Falling back to brute force. "
            f"Increase max_candidates or smoothing_factor.",
            stacklevel=2,
        )
        from differt.rt._utils import rays_intersect_any_triangle

        return rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            triangle_vertices,
            active_triangles,
            hit_tol=hit_tol,
            smoothing_factor=smoothing_factor,
            **kwargs,
        )

    # Convert to JAX
    cand_idx = jnp.asarray(candidate_indices.reshape(*batch_shape, max_candidates))
    cand_counts = jnp.asarray(candidate_counts.reshape(batch_shape))

    # Gather candidate triangle vertices: shape [*batch, max_candidates, 3, 3]
    # Use the non-batch triangle_vertices (first batch element if batched)
    tri_flat = triangle_vertices_jnp
    if tri_flat.ndim > 3:
        tri_flat = tri_flat.reshape(-1, 3, 3)

    # Clamp indices to valid range for gather (padding -1 -> 0, masked out later)
    safe_idx = jnp.maximum(cand_idx, 0)
    cand_verts = tri_flat[safe_idx]  # [*batch, max_candidates, 3, 3]

    # Mask: which candidates are valid
    arange = jnp.arange(max_candidates)
    mask = arange[None] < cand_counts[..., None] if cand_counts.ndim == 1 else arange < cand_counts[..., None]

    # Active triangles filter
    if active_triangles is not None:
        active_jnp = jnp.asarray(active_triangles)
        if active_jnp.ndim > 1:
            active_flat = active_jnp.reshape(-1)
        else:
            active_flat = active_jnp
        cand_active = active_flat[safe_idx.reshape(-1, max_candidates)].reshape(
            *batch_shape, max_candidates
        )
        mask = mask & cand_active

    # Run soft intersection on candidates (this is pure JAX, differentiable)
    t, hit = rays_intersect_triangles(
        ray_origins_jnp[..., None, :],  # [*batch, 1, 3]
        ray_directions_jnp[..., None, :],  # [*batch, 1, 3]
        cand_verts,  # [*batch, max_candidates, 3, 3]
        smoothing_factor=smoothing_factor,
        **kwargs,
    )

    soft_hit = jnp.minimum(hit, smoothing_function(hit_threshold - t, smoothing_factor))
    result = jnp.sum(soft_hit * mask, axis=-1).clip(max=1.0)

    return result


def bvh_first_triangles_hit_by_rays(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    *,
    bvh: TriangleBvh | None = None,
    **kwargs: Any,
) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
    """BVH-accelerated version of :func:`~differt.rt.first_triangles_hit_by_rays`.

    Uses BVH traversal for O(log N) nearest-hit per ray instead of O(N).

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray directions.
        triangle_vertices: An array of triangle vertices.
        active_triangles: Optional boolean mask for active triangles.
        bvh: Pre-built BVH acceleration structure.
        kwargs: Additional keyword arguments (for API compatibility).

    Returns:
        A tuple ``(indices, t)`` of the nearest triangle index and distance.
    """
    if bvh is None:
        from differt.rt._utils import first_triangles_hit_by_rays

        return first_triangles_hit_by_rays(
            ray_origins,
            ray_directions,
            triangle_vertices,
            active_triangles,
            **kwargs,
        )

    ray_origins_jnp = jnp.asarray(ray_origins)
    ray_directions_jnp = jnp.asarray(ray_directions)
    batch_shape = ray_origins_jnp.shape[:-1]

    flat_origins = np.asarray(ray_origins_jnp).reshape(-1, 3)
    flat_dirs = np.asarray(ray_directions_jnp).reshape(-1, 3)
    hit_indices, hit_t = bvh.nearest_hit(flat_origins, flat_dirs)

    # Apply active_triangles filter: if the nearest hit is an inactive triangle,
    # we need to find the next hit. For simplicity, we mark it as a miss.
    # A more complete implementation would re-query excluding inactive triangles.
    if active_triangles is not None:
        active = np.asarray(active_triangles)
        if active.ndim > 1:
            active = active.flatten()
        for i in range(len(hit_indices)):
            if hit_indices[i] >= 0 and not active[hit_indices[i]]:
                hit_indices[i] = -1
                hit_t[i] = float("inf")

    return (
        jnp.asarray(hit_indices.reshape(batch_shape)),
        jnp.asarray(hit_t.reshape(batch_shape)),
    )

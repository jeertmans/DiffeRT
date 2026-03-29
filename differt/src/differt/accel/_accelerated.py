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
    "bvh_triangles_visible_from_vertices",
)

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from differt.accel._bvh import TriangleBvh, compute_expansion_radius
from differt.rt._utils import rays_intersect_triangles
from differt.utils import smoothing_function

if TYPE_CHECKING:
    from jaxtyping import Array, ArrayLike, Bool, Float, Int


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
        from differt.rt._utils import rays_intersect_any_triangle  # noqa: PLC0415

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
        # Pass active_triangles mask directly to Rust BVH so it skips
        # inactive triangles and finds the nearest *active* hit.
        flat_origins = np.asarray(ray_origins_jnp).reshape(-1, 3)
        flat_dirs = np.asarray(ray_directions_jnp).reshape(-1, 3)
        mask_np = None
        if active_triangles is not None:
            mask_np = np.ascontiguousarray(np.asarray(active_triangles).flatten())
        hit_indices, hit_t = bvh.nearest_hit(flat_origins, flat_dirs, mask_np)

        # Apply hit_threshold: only count hits with t < hit_threshold
        any_hit = (hit_indices >= 0) & (hit_t < float(hit_threshold))

        return jnp.asarray(any_hit.reshape(batch_shape))

    # Soft/differentiable mode: BVH candidate selection + JAX soft intersection
    alpha = float(smoothing_factor)

    # Estimate triangle size for expansion radius
    tri_np = np.asarray(triangle_vertices_jnp)
    flat_tri = tri_np.reshape(-1, 3, 3) if tri_np.ndim > 3 else tri_np  # noqa: PLR2004
    # Use mean edge length as characteristic size
    edges = np.diff(flat_tri, axis=-2, append=flat_tri[..., :1, :])
    mean_tri_size = float(np.mean(np.linalg.norm(edges, axis=-1)))
    expansion = compute_expansion_radius(alpha, mean_tri_size, epsilon_grad)

    # Check if expansion is too large (soft smoothing -> fallback to brute force)
    scene_diag = float(
        np.linalg.norm(
            flat_tri.reshape(-1, 3).max(axis=0) - flat_tri.reshape(-1, 3).min(axis=0)
        )
    )
    if expansion > scene_diag:
        from differt.rt._utils import rays_intersect_any_triangle  # noqa: PLC0415

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
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"BVH candidate count ({int(candidate_counts.max())}) exceeds "
            f"max_candidates ({max_candidates}). Falling back to brute force. "
            f"Increase max_candidates or smoothing_factor.",
            stacklevel=2,
        )
        from differt.rt._utils import rays_intersect_any_triangle  # noqa: PLC0415

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
    if tri_flat.ndim > 3:  # noqa: PLR2004
        tri_flat = tri_flat.reshape(-1, 3, 3)

    # Clamp indices to valid range for gather (padding -1 -> 0, masked out later)
    safe_idx = jnp.maximum(cand_idx, 0)
    cand_verts = tri_flat[safe_idx]  # [*batch, max_candidates, 3, 3]

    # Mask: which candidates are valid
    arange = jnp.arange(max_candidates)
    mask = (
        arange[None] < cand_counts[..., None]
        if cand_counts.ndim == 1
        else arange < cand_counts[..., None]
    )

    # Active triangles filter
    if active_triangles is not None:
        active_jnp = jnp.asarray(active_triangles)
        active_flat = active_jnp.reshape(-1) if active_jnp.ndim > 1 else active_jnp
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
    return jnp.sum(soft_hit * mask, axis=-1).clip(max=1.0)


def bvh_triangles_visible_from_vertices(
    vertices: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    num_rays: int = int(1e6),
    *,
    bvh: TriangleBvh | None = None,
    **kwargs: Any,
) -> Bool[Array, "*batch num_triangles"]:
    """BVH-accelerated version of :func:`~differt.rt.triangles_visible_from_vertices`.

    Uses BVH nearest-hit for O(log N) per ray instead of O(N), avoiding JAX's
    O(rays * triangles) memory allocation.

    Args:
        vertices: An array of vertices, used as origins of the rays.
        triangle_vertices: An array of triangle vertices.
        active_triangles: An optional boolean mask for active triangles.
        num_rays: The number of rays to launch per vertex.
        bvh: Pre-built BVH acceleration structure.
        kwargs: Additional keyword arguments (for API compatibility).

    Returns:
        For each triangle, whether it is visible from any of the rays.
    """
    if bvh is None:
        from differt.rt._utils import triangles_visible_from_vertices  # noqa: PLC0415

        return triangles_visible_from_vertices(
            vertices,
            triangle_vertices,
            active_triangles,
            num_rays=num_rays,
            **kwargs,
        )

    vertices_jnp = jnp.asarray(vertices)
    triangle_vertices_jnp = jnp.asarray(triangle_vertices)
    num_triangles = triangle_vertices_jnp.shape[-3]

    # Compute viewing frustum and generate fibonacci lattice directions
    from differt.geometry import fibonacci_lattice, viewing_frustum  # noqa: PLC0415

    triangle_centers = triangle_vertices_jnp.mean(axis=-2, keepdims=True)
    world_vertices = jnp.concat(
        (triangle_vertices_jnp, triangle_centers), axis=-2
    ).reshape(*triangle_vertices_jnp.shape[:-3], -1, 3)

    if active_triangles is not None:
        active_jnp = jnp.asarray(active_triangles)
        active_vertices = jnp.repeat(active_jnp, 4, axis=-1)
    else:
        active_vertices = None

    ray_origins = vertices_jnp

    frustum = viewing_frustum(
        ray_origins,
        world_vertices,
        active_vertices=active_vertices,
    )

    ray_directions = jnp.vectorize(
        lambda n, frustum: fibonacci_lattice(n, frustum=frustum),
        excluded={0},
        signature="(2,3)->(n,3)",
    )(num_rays, frustum)

    # Flatten batch dims for BVH queries
    batch_shape = ray_origins.shape[:-1]
    flat_origins = np.asarray(ray_origins).reshape(-1, 3)
    flat_dirs = np.asarray(ray_directions).reshape(-1, num_rays, 3)
    num_vertices = flat_origins.shape[0]

    # Tile origins and flatten: each origin gets num_rays copies
    # Shape: (num_vertices * num_rays, 3)
    all_origins = np.repeat(flat_origins, num_rays, axis=0)
    all_dirs = flat_dirs.reshape(-1, 3)

    # Ensure contiguous for Rust
    all_origins = np.ascontiguousarray(all_origins)
    all_dirs = np.ascontiguousarray(all_dirs)

    # Single BVH call for all rays
    hit_indices, _ = bvh.nearest_hit(all_origins, all_dirs)

    # Reshape to (num_vertices, num_rays)
    hit_indices = hit_indices.reshape(num_vertices, num_rays)

    # Build visibility mask
    visible = np.zeros((*batch_shape, num_triangles), dtype=bool)
    flat_visible = visible.reshape(num_vertices, num_triangles)

    active_np = None
    if active_triangles is not None:
        active_np = np.asarray(active_triangles)
        if active_np.ndim > 1:
            active_np = active_np.reshape(-1)

    for i in range(num_vertices):
        valid = hit_indices[i] >= 0
        valid_indices = hit_indices[i][valid]
        if active_np is not None:
            active_mask = active_np[valid_indices]
            valid_indices = valid_indices[active_mask]
        unique_hits = np.unique(valid_indices)
        flat_visible[i, unique_hits] = True

    return jnp.asarray(visible)


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
        from differt.rt._utils import first_triangles_hit_by_rays  # noqa: PLC0415

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

    # Pass active_triangles mask directly to Rust BVH so it skips inactive
    # triangles during traversal and finds the nearest *active* hit.
    mask_np = None
    if active_triangles is not None:
        mask_np = np.ascontiguousarray(np.asarray(active_triangles).flatten())
    hit_indices, hit_t = bvh.nearest_hit(flat_origins, flat_dirs, mask_np)

    return (
        jnp.asarray(hit_indices.reshape(batch_shape)),
        jnp.asarray(hit_t.reshape(batch_shape)),
    )

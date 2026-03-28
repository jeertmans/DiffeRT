"""JAX FFI wrappers for BVH acceleration.

These functions call into Rust BVH queries via XLA FFI, enabling BVH
operations inside JIT-compiled JAX functions (``jax.jit``, ``jax.lax.scan``).

Requires ``differt-core`` built with the ``xla-ffi`` feature.
"""

from __future__ import annotations

__all__ = (
    "ffi_nearest_hit",
    "ffi_get_candidates",
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

_FFI_REGISTERED = False


def _ensure_registered():
    """Register BVH FFI targets with JAX (once)."""
    global _FFI_REGISTERED
    if _FFI_REGISTERED:
        return

    try:
        from differt_core import _differt_core

        bvh_mod = _differt_core.accel.bvh
        bvh_nearest_hit_capsule = bvh_mod.bvh_nearest_hit_capsule
        bvh_get_candidates_capsule = bvh_mod.bvh_get_candidates_capsule
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "BVH XLA FFI not available. Rebuild differt-core with "
            "the xla-ffi feature: "
            "PYTHON_SYS_EXECUTABLE=$(which python) "
            "maturin develop --strip"
        ) from e

    jax.ffi.register_ffi_target(
        "bvh_nearest_hit", bvh_nearest_hit_capsule(), platform="cpu"
    )
    jax.ffi.register_ffi_target(
        "bvh_get_candidates", bvh_get_candidates_capsule(), platform="cpu"
    )
    _FFI_REGISTERED = True


def ffi_nearest_hit(
    ray_origins: Float[Array, "num_rays 3"],
    ray_directions: Float[Array, "num_rays 3"],
    *,
    bvh_id: int,
    active_mask: Array | None = None,
):
    """BVH nearest-hit via XLA FFI. Works inside ``jax.jit``.

    Args:
        ray_origins: Ray origins with shape ``(num_rays, 3)``.
        ray_directions: Ray directions with shape ``(num_rays, 3)``.
        bvh_id: Registry ID from ``bvh.register()``.
        active_mask: Optional boolean mask with shape ``(num_triangles,)``.
            When provided, only triangles where the mask is ``True`` are
            considered during traversal, correctly finding the nearest
            *active* hit.

    Returns:
        A tuple ``(hit_indices, hit_t)`` with triangle index (``-1`` for miss)
        and parametric distance.
    """
    _ensure_registered()

    num_rays = ray_origins.shape[0]
    out_types = [
        jax.ShapeDtypeStruct((num_rays,), jnp.int32),  # hit_indices
        jax.ShapeDtypeStruct((num_rays,), jnp.float32),  # hit_t
    ]

    call = jax.ffi.ffi_call(
        "bvh_nearest_hit",
        out_types,
        vmap_method="broadcast_all",
    )

    # Pass active_mask as a PRED buffer; empty array means no mask
    if active_mask is None:
        mask_buf = jnp.empty((0,), dtype=jnp.bool_)
    else:
        mask_buf = active_mask.astype(jnp.bool_)

    return call(
        ray_origins.astype(jnp.float32),
        ray_directions.astype(jnp.float32),
        mask_buf,
        bvh_id=np.uint64(bvh_id),
    )


def ffi_get_candidates(
    ray_origins: Float[Array, "num_rays 3"],
    ray_directions: Float[Array, "num_rays 3"],
    *,
    bvh_id: int,
    expansion: float = 0.0,
    max_candidates: int = 256,
):
    """BVH candidate selection via XLA FFI. Works inside ``jax.jit``.

    Args:
        ray_origins: Ray origins with shape ``(num_rays, 3)``.
        ray_directions: Ray directions with shape ``(num_rays, 3)``.
        bvh_id: Registry ID from ``bvh.register()``.
        expansion: Bounding box expansion for differentiable mode.
        max_candidates: Maximum candidates per ray.

    Returns:
        A tuple ``(candidate_indices, candidate_counts)`` where indices
        are padded with ``-1`` and counts indicate valid entries.
    """
    _ensure_registered()

    num_rays = ray_origins.shape[0]
    out_types = [
        jax.ShapeDtypeStruct(
            (num_rays, max_candidates), jnp.int32
        ),  # candidate_indices
        jax.ShapeDtypeStruct((num_rays,), jnp.int32),  # candidate_counts
    ]

    call = jax.ffi.ffi_call(
        "bvh_get_candidates",
        out_types,
        vmap_method="broadcast_all",
    )

    return call(
        ray_origins.astype(jnp.float32),
        ray_directions.astype(jnp.float32),
        bvh_id=np.uint64(bvh_id),
        expansion=np.float32(expansion),
        max_candidates=np.int32(max_candidates),
    )

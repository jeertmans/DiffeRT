__all__ = ("TriangleBvh",)

import math

import numpy as np
from jaxtyping import ArrayLike, Float

from differt_core.accel.bvh import TriangleBvh as _RustBvh


class TriangleBvh:
    """BVH acceleration structure for triangle meshes.

    Builds a SAH-based Bounding Volume Hierarchy over triangle vertices.
    Supports two query types:

    - :meth:`nearest_hit`: find the closest triangle per ray (for SBR)
    - :meth:`get_candidates`: find candidate triangles per ray with expanded
      bounding boxes (for differentiable mode)

    Args:
        triangle_vertices: Triangle vertices.

    Example:
        >>> import jax.numpy as jnp
        >>> from differt.accel.bvh import TriangleBvh
        >>> verts = jnp.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=jnp.float32)
        >>> bvh = TriangleBvh(verts)
        >>> bvh.num_triangles
        1
    """

    def __init__(
        self, triangle_vertices: Float[ArrayLike, "num_triangles 3 3"]
    ) -> None:
        # TODO: why would we pass 2-dimension input?
        verts = np.asarray(triangle_vertices, dtype=np.float32)
        if verts.ndim == 3:  # noqa: PLR2004
            # Shape (num_triangles, 3, 3) -> (num_triangles * 3, 3)
            verts = verts.reshape(-1, 3)
        self._inner = _RustBvh(verts)

    @property
    def num_triangles(self) -> int:
        """Number of triangles in the BVH."""
        return self._inner.num_triangles

    @property
    def num_nodes(self) -> int:
        """Number of BVH nodes used."""
        return self._inner.num_nodes

    def nearest_hit(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        active_mask: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the nearest active triangle hit by each ray.

        Args:
            ray_origins: Ray origins with shape ``(num_rays, 3)``.
            ray_directions: Ray directions with shape ``(num_rays, 3)``.
            active_mask: Optional boolean mask with shape ``(num_triangles,)``.
                When provided, only triangles where the mask is ``True`` are
                considered during traversal.

        Returns:
            A tuple ``(hit_indices, hit_t)`` where ``hit_indices`` has
            shape ``(num_rays,)`` with triangle index (``-1`` for miss)
            and ``hit_t`` has shape ``(num_rays,)`` with parametric distance.

        Example:
            >>> import jax.numpy as jnp
            >>> from differt.accel.bvh import TriangleBvh
            >>> verts = jnp.array(
            ...     [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=jnp.float32
            ... )
            >>> bvh = TriangleBvh(verts)
            >>> origins = jnp.array([[0.1, 0.1, 1.0]])
            >>> dirs = jnp.array([[0.0, 0.0, -1.0]])
            >>> idx, t = bvh.nearest_hit(origins, dirs)
            >>> int(idx[0])
            0
        """
        origins = np.asarray(ray_origins, dtype=np.float32)
        dirs = np.asarray(ray_directions, dtype=np.float32)
        mask = None
        if active_mask is not None:
            mask = np.ascontiguousarray(np.asarray(active_mask).flatten())
        if origins.ndim > 2:  # noqa: PLR2004
            orig_shape = origins.shape[:-1]
            origins = origins.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)
            idx, t = self._inner.nearest_hit(origins, dirs, mask)
            return idx.reshape(orig_shape), t.reshape(orig_shape)
        return self._inner.nearest_hit(origins, dirs, mask)

    def register(self) -> int:
        """Register this BVH for XLA FFI access.

        Returns:
            Integer ID for use with JAX FFI handlers.

        Example:
            >>> import jax.numpy as jnp
            >>> from differt.accel.bvh import TriangleBvh
            >>> verts = jnp.array(
            ...     [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=jnp.float32
            ... )
            >>> bvh = TriangleBvh(verts)
            >>> bvh_id = bvh.register()
            >>> bvh_id > 0
            True
            >>> bvh.unregister()
        """
        return self._inner.register()

    def unregister(self) -> None:
        """Remove this BVH from the global registry."""
        self._inner.unregister()

    def get_candidates(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        expansion: float = 0.0,
        max_candidates: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find candidate triangles whose expanded AABBs intersect each ray.

        For differentiable mode, the expansion captures all triangles with
        non-negligible gradient contribution.

        Args:
            ray_origins: Ray origins with shape ``(num_rays, 3)``.
            ray_directions: Ray directions with shape ``(num_rays, 3)``.
            expansion: Bounding box expansion amount.
            max_candidates: Maximum candidates per ray.

        Returns:
            A tuple ``(candidate_indices, candidate_counts)`` where
            ``candidate_indices`` has shape ``(num_rays, max_candidates)``
            padded with ``-1``, and ``candidate_counts`` has shape ``(num_rays,)``.

        Example:
            >>> import jax.numpy as jnp
            >>> from differt.accel.bvh import TriangleBvh
            >>> verts = jnp.array(
            ...     [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=jnp.float32
            ... )
            >>> bvh = TriangleBvh(verts)
            >>> origins = jnp.array([[0.1, 0.1, 1.0]])
            >>> dirs = jnp.array([[0.0, 0.0, -1.0]])
            >>> idx, counts = bvh.get_candidates(origins, dirs, expansion=0.0)
            >>> int(counts[0]) >= 1
            True
        """
        origins = np.asarray(ray_origins, dtype=np.float32)
        dirs = np.asarray(ray_directions, dtype=np.float32)
        if origins.ndim > 2:  # noqa: PLR2004
            orig_shape = origins.shape[:-1]
            origins = origins.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)
            idx, counts = self._inner.get_candidates(
                origins, dirs, expansion, max_candidates
            )
            return (
                idx.reshape(*orig_shape, max_candidates),
                counts.reshape(orig_shape),
            )
        return self._inner.get_candidates(origins, dirs, expansion, max_candidates)


def compute_expansion_radius(
    smoothing_factor: float,
    triangle_size: float = 1.0,
    epsilon_grad: float = 1e-7,
) -> float:
    """Compute BVH expansion radius for differentiable mode.

    The expansion guarantees that all triangles with gradient contribution
    above ``epsilon_grad`` are included in the candidate set.

    Args:
        smoothing_factor: The smoothing parameter (alpha).
        triangle_size: Approximate characteristic triangle size.
        epsilon_grad: Gradient truncation threshold.

    Returns:
        The expansion radius in the same units as triangle_size.

    Example:
        >>> from differt.accel.bvh import compute_expansion_radius
        >>> r = compute_expansion_radius(10.0, triangle_size=1.0)
        >>> r > 0
        True
    """
    if smoothing_factor <= 0:
        return float("inf")
    return triangle_size * math.log(1.0 / epsilon_grad) / smoothing_factor

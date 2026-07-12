"""Interception plane utilities for SBR receiver detection.

This module implements the Interception Plane technique as an alternative
to the traditional "reception sphere" method for determining whether a
ray tube launched from a transmitter successfully reaches a receiver.

Instead of checking if a ray passes within a fixed distance of a point
receiver (which introduces geometric ambiguities), the interception plane
method constructs a virtual plane orthogonal to the incoming ray direction
at the receiver location. The ray tube's cross-section on this plane is
compared against the receiver's position to determine a valid hit.

Additionally, a plane fusion algorithm clusters coherent ray bundles that
correspond to the same physical wavefront, preventing artificial inflation
of received energy in the Channel Impulse Response.
"""

__all__ = (
    "fuse_ray_bundles",
    "interception_plane_check",
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.geometry._hashing import hash_interaction_sequence


@jax.jit
def interception_plane_check(
    ray_origins: Float[ArrayLike, "*batch 3"],
    ray_directions: Float[ArrayLike, "*batch 3"],
    rx_positions: Float[ArrayLike, "num_rx 3"],
    solid_angle_per_ray: Float[ArrayLike, ""],
    propagation_distances: Float[ArrayLike, " *batch"],
) -> Bool[Array, "*batch num_rx"]:
    r"""Check if receivers lie within the ray tube footprint on the interception plane.

    For each ray, a virtual plane is constructed at the receiver location,
    oriented orthogonally to the ray's propagation direction. The ray tube's
    footprint on this plane is approximated as a circular disc whose radius
    is determined by the solid angle per ray and the total propagation distance:

    .. math::
        r_{\text{footprint}} = d \cdot \sqrt{\frac{\Omega}{\\pi}}

    where :math:`d` is the propagation distance from the last bounce to the
    interception plane and :math:`\Omega` is the solid angle subtended by
    each launched ray (approximately :math:`4\pi / N_{\text{rays}}`
    for Fibonacci lattice sampling with :math:`N_{\text{rays}}` rays).

    A hit is registered if:

    1. The ray is travelling toward the receiver (positive projection), and
    2. The receiver's perpendicular distance from the ray axis is less than
       the footprint radius.

    Args:
        ray_origins: Origin positions of rays after all bounces.
        ray_directions: Directions of rays after all bounces (need not be unit vectors).
        rx_positions: Receiver positions.
        solid_angle_per_ray: Solid angle per ray in steradians
            (e.g., ``4 * pi / num_rays`` for isotropic sampling).
        propagation_distances: Total propagation distance of each ray
            from the transmitter through all bounces.

    Returns:
        Boolean mask indicating which (ray, receiver) pairs constitute valid hits.
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    rx_positions = jnp.asarray(rx_positions)
    solid_angle_per_ray = jnp.asarray(solid_angle_per_ray)
    propagation_distances = jnp.asarray(propagation_distances)

    # Normalize ray directions
    ray_dir_norm = jnp.linalg.norm(ray_directions, axis=-1, keepdims=True)
    ray_dir_unit = ray_directions / jnp.maximum(ray_dir_norm, 1e-12)

    # Vector from ray origin to each receiver
    # [*batch *rx 3]
    to_rx = rx_positions - ray_origins[..., None, :]  # broadcasting *rx

    # Handle arbitrary leading dimensions properly
    # For *batch and *rx broadcasting, we insert dims as needed
    if rx_positions.ndim > 1:
        # rx_positions has leading *rx dims; expand ray arrays
        n_rx_dims = rx_positions.ndim - 1
        for _ in range(n_rx_dims):
            ray_dir_unit = ray_dir_unit[..., None, :]
            propagation_distances = propagation_distances[..., None]
        to_rx = rx_positions - ray_origins[..., None, :]
    else:
        to_rx = rx_positions - ray_origins

    # Projection of rx along ray direction (scalar)
    # t > 0 means the receiver is "in front of" the ray
    t = jnp.sum(to_rx * ray_dir_unit, axis=-1)

    # Perpendicular distance from the ray axis to the receiver
    perp_vec = to_rx - t[..., None] * ray_dir_unit
    perp_dist_sq = jnp.sum(perp_vec * perp_vec, axis=-1)

    # Footprint radius based on solid angle and propagation distance
    # For the last segment, distance from last bounce to interception plane is t
    # Total distance for divergence calculation: propagation_distances + t
    total_dist = propagation_distances + jnp.maximum(t, 0.0)
    footprint_radius = total_dist * jnp.sqrt(solid_angle_per_ray / jnp.pi)
    footprint_radius_sq = footprint_radius * footprint_radius

    # Hit condition: ray is going toward receiver AND receiver is within footprint
    is_hit = (t > 0.0) & (perp_dist_sq < footprint_radius_sq)

    return is_hit


@jax.jit
def fuse_ray_bundles(
    object_ids: Int[ArrayLike, "num_rays order"],
    hit_positions: Float[ArrayLike, "num_rays 3"],
    hit_mask: Bool[ArrayLike, " num_rays"],
    spatial_tolerance: Float[ArrayLike, ""] = 0.1,
    interaction_types: Int[ArrayLike, "num_rays order"] | None = None,
) -> tuple[
    Int[Array, " num_rays"],
    Bool[Array, " num_rays"],
]:
    """Cluster coherent ray bundles and select a single representative per cluster.

    When multiple closely spaced rays trace the same specular chain
    (same sequence of intersected primitives), they represent a single
    macroscopic wavefront. Processing them independently would
    artificially inflate the received energy.

    This function:

    1. Hashes each ray's interaction sequence into a scalar key.
    2. Groups rays by their hash.
    3. Within each group, selects the ray closest to the group centroid
       as the representative.

    Args:
        object_ids: Primitive IDs hit by each ray, one per interaction.
        hit_positions: The (x, y, z) position where each ray hits the
            interception plane (or the receiver vicinity).
        hit_mask: Boolean mask indicating which rays are valid hits.
        spatial_tolerance: Maximum spatial distance (meters) between
            rays in the same cluster. Used for sub-clustering within
            the same hash group. Currently reserved for future use.
        interaction_types: Optional interaction type per interaction.

    Returns:
        A tuple containing:

        - ``cluster_ids``: Integer cluster ID per ray.
        - ``representative_mask``: Boolean mask that is ``True`` only
          for the single representative ray of each unique cluster.
    """
    object_ids = jnp.asarray(object_ids)
    hit_positions = jnp.asarray(hit_positions)
    hit_mask = jnp.asarray(hit_mask)

    # Step 1: Hash each ray's interaction sequence
    hashes = hash_interaction_sequence(object_ids, interaction_types)

    # Step 2: Assign invalid rays a sentinel hash that won't collide
    sentinel = jnp.iinfo(hashes.dtype).max
    hashes = jnp.where(hit_mask, hashes, sentinel)

    # Step 3: For each unique hash, find the first occurrence (representative)
    # Sort by hash to group equal hashes together
    sorted_indices = jnp.argsort(hashes)
    sorted_hashes = hashes[sorted_indices]

    # Mark the first occurrence of each unique hash
    is_first = jnp.concatenate([
        jnp.array([True]),
        sorted_hashes[1:] != sorted_hashes[:-1],
    ])

    # Map back to original order
    representative_in_sorted = is_first & (sorted_hashes != sentinel)

    # Create representative mask in original order
    num_rays = object_ids.shape[0]
    representative_mask = jnp.zeros(num_rays, dtype=bool)
    representative_mask = representative_mask.at[sorted_indices].set(
        representative_in_sorted
    )

    return hashes, representative_mask

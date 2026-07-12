"""Hashing utilities for specular chain deduplication.

This module provides deterministic, collision-free hashing functions
for encoding ordered sequences of primitive IDs (and optionally interaction
types) into unique integer keys. These are used by the SBR pipeline to
avoid redundant electromagnetic field computations for rays that trace
the same sequence of geometric primitives.
"""

__all__ = (
    "cantor_pair",
    "hash_interaction_sequence",
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int


@jax.jit
def cantor_pair(
    a: Int[Array, " *batch"],
    b: Int[Array, " *batch"],
) -> Int[Array, " *batch"]:
    r"""Map two non-negative integers to a unique non-negative integer.

    Uses Cantor's pairing function:

    .. math::
        \pi(a, b) = \frac{(a + b)(a + b + 1)}{2} + b

    This is a bijection from :math:`\mathbb{N}^2 \to \mathbb{N}`,
    guaranteeing collision-free encoding.

    Args:
        a: First integer array.
        b: Second integer array.

    Returns:
        The Cantor-paired integer array.

    Examples:
        >>> from differt.geometry._hashing import cantor_pair
        >>>
        >>> a = jnp.array([0, 1, 2])
        >>> b = jnp.array([0, 0, 1])
        >>> cantor_pair(a, b)
        Array([0, 1, 7], dtype=int32)
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    s = a + b
    return (s * (s + 1)) // 2 + b


@jax.jit
def hash_interaction_sequence(
    object_ids: Int[Array, "*batch order"],
    interaction_types: Int[Array, "*batch order"] | None = None,
) -> Int[Array, " *batch"]:
    """Map an ordered sequence of interacted primitive IDs to a unique hash.

    If ``interaction_types`` is provided, the hash also encodes the
    type of each interaction (e.g., reflection, diffraction, transmission),
    ensuring that two paths with the same primitive sequence but different
    interaction types produce distinct hashes.

    The hash is computed by iteratively applying :func:`cantor_pair`
    to fold the sequence into a single scalar per batch element.

    Args:
        object_ids: Array of primitive (triangle/quad) indices,
            one per interaction along the path.
        interaction_types: Optional array of interaction type indices,
            with the same shape as ``object_ids``.

    Returns:
        A scalar hash per batch element.

    Examples:
        >>> from differt.geometry._hashing import hash_interaction_sequence
        >>>
        >>> # Two paths hitting the same triangles in different order produce different hashes
        >>> ids_a = jnp.array([[0, 1, 2]])
        >>> ids_b = jnp.array([[2, 1, 0]])
        >>> hash_a = hash_interaction_sequence(ids_a)
        >>> hash_b = hash_interaction_sequence(ids_b)
        >>> bool(hash_a != hash_b)
        True
    """
    object_ids = jnp.asarray(object_ids)

    if interaction_types is not None:
        interaction_types = jnp.asarray(interaction_types)
        # Encode (object_id, interaction_type) pair at each step
        combined = cantor_pair(object_ids, interaction_types)
    else:
        combined = object_ids

    # Fold over the interaction axis using cantor_pair as the combining function
    def scan_fun(
        carry: Int[Array, " *batch"],
        element: Int[Array, " *batch"],
    ) -> tuple[Int[Array, " *batch"], None]:
        return cantor_pair(carry, element), None

    # Transpose to iterate over the 'order' axis
    # combined: [*batch order] -> transposed: [order *batch]
    *batch, order = combined.shape
    elements = jnp.moveaxis(combined, -1, 0)

    # Initialize with a fixed seed to differentiate from raw object IDs
    init = jnp.full(batch, 0, dtype=combined.dtype) if batch else jnp.int32(0)

    result, _ = jax.lax.scan(scan_fun, init, elements)
    return result

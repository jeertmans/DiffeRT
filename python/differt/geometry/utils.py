"""Utilities for working with 3D geometries."""
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def pairwise_cross(
    u: Float[Array, "m 3"], v: Float[Array, "n 3"]
) -> Float[Array, "m n 3"]:
    """
    Compute the pairwise cross product between two arrays of vectors.

    Args:
        u: First array of vectors.
        v: Second array of vectors.

    Returns:
        A 3D tensor with all cross products.
    """
    return jnp.cross(u[:, None, :], v[None, :, :])


@jaxtyped(typechecker=typechecker)
def normalize(
    vector: Float[Array, "*batch 3"],
) -> tuple[Float[Array, "*batch 3"], Float[Array, " *batch"]]:
    """
    Normalize vectors and also return their length.

    Args:
        vector: An array of vectors.

    Returns:
        The normalized vector and their length.

    :Examples:

    >>> from differt.geometry.utils import (
    ...     normalize,
    ... )
    >>>
    >>> vector = jnp.array([1.0, 1.0, 1.0])
    >>> normalize(vector)  # [1., 1., 1.] / sqrt(3), sqrt(3)
    (Array([0.57735026, 0.57735026, 0.57735026], dtype=float32),
     Array(1.7320508, dtype=float32))
    >>> zero = jnp.array([0.0, 0.0, 0.0])
    >>> normalize(zero)  # Special behavior at 0.
    (Array([0., 0., 0.], dtype=float32), Array(1., dtype=float32))
    """
    length: Array = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    length = jnp.where(length == 0.0, jnp.ones_like(length), length)

    return vector / length, jnp.squeeze(length, axis=-1)

"""Utilities for working with 3D geometries."""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=typechecker)
def pairwise_cross(
    u: Float[Array, "m 3"], v: Float[Array, "n 3"]
) -> Float[Array, "m n 3"]:
    """
    Compute the pairwise cross product between two arrays of vectors.

    Args:
        u: First array of vectors.
        v: Second array of vectors.

    Return:
        A 3D tensor with all cross products.
    """
    return jnp.cross(u[:, None, :], v[None, :, :])


@jax.jit
@jaxtyped(typechecker=typechecker)
def normalize(
    vector: Float[Array, "*batch 3"],
) -> tuple[Float[Array, "*batch 3"], Float[Array, " *batch"]]:
    """
    Normalize vectors and also return their length.

    This function avoids division by zero by checking vectors
    with zero-length, and returning unit length instead.

    Args:
        vector: An array of vectors.

    Return:
        The normalized vector and their length.

    Examples:
        >>> from differt.geometry.utils import (
        ...     normalize,
        ... )
        >>>
        >>> vector = jnp.array([1.0, 1.0, 1.0])
        >>> normalize(vector)  # [1., 1., 1.] / sqrt(3), sqrt(3)
        (Array([0.5773503, 0.5773503, 0.5773503], dtype=float32),
         Array(1.7320508, dtype=float32))
        >>> zero = jnp.array([0.0, 0.0, 0.0])
        >>> normalize(zero)  # Special behavior at 0.
        (Array([0., 0., 0.], dtype=float32), Array(1., dtype=float32))
    """
    length: Array = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    length = jnp.where(length == 0.0, jnp.ones_like(length), length)

    return vector / length, jnp.squeeze(length, axis=-1)


@partial(jax.jit, static_argnames=("normalize",))
@jaxtyped(typechecker=typechecker)
def orthogonal_basis(
    u: Float[Array, "*batch 3"], normalize: bool = True
) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """
    Generate ``v`` and ``w``, two other arrays of unit vectors that form with input ``u`` an orthogonal basis.

    Args:
        u: The first direction of the orthogonal basis.
            It must have a unit length.
        normalize: Whether the output vectors should be normalized.

            This may be needed, especially for vector ``v``,
            as floating-point error can accumulate so much
            that the vector lengths may diverge from the unit
            length by 10% or even more!

    Return:
        A pair of unit vectors, ``v`` and ``w``.

    Examples:
        >>> from differt.geometry.utils import (
        ...     normalize,
        ...     orthogonal_basis,
        ... )
        >>>
        >>> u = jnp.array([1.0, 0.0, 0.0])
        >>> orthogonal_basis(u)
        (Array([ 0., -1.,  0.], dtype=float32), Array([ 0.,  0., -1.], dtype=float32))
        >>> u, _ = normalize(jnp.array([1.0, 1.0, 1.0]))
        >>> orthogonal_basis(u)
        (Array([ 0.4082483, -0.8164966,  0.4082483], dtype=float32),
         Array([ 0.7071068,  0.       , -0.7071068], dtype=float32))
    """
    vp = jnp.stack((u[..., 2], -u[..., 0], u[..., 1]), axis=-1)
    w = jnp.cross(u, vp, axis=-1)
    v = jnp.cross(w, u, axis=-1)

    if normalize:
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)

    return v, w

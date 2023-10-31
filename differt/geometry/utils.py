import jax.numpy as jnp

from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def pairwise_cross(
    u: Float[Array, "m 3"], v: Float[Array, "n 3"]
) -> Float[Array, "m n 3"]:
    """
    Compute the pairwise cross product between two arrays of vectors.
    """
    return jnp.cross(u[:, None, :], v[None, :, :])

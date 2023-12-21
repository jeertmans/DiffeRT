import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def pairwise_cross(
    u: Float[Array, "m 3"], v: Float[Array, "n 3"]
) -> Float[Array, "m n 3"]:
    """
    Compute the pairwise cross product between two arrays of vectors.

    :param: The first array of vectors.
    :param: The second array of vectors.
    :return: A 3D tensor with all cross products.
    """
    return jnp.cross(u[:, None, :], v[None, :, :])

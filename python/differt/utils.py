"""General purpose utilities."""
import jax.numpy as jnp
from jaxtyping import Array, Shaped, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def sorted_array2(array: Shaped[Array, "m n"]) -> Shaped[Array, "m n"]:
    """
    Sort a 2D array by row and by colunm.

    Args:
        array: The input array.

    Returns:
        A sorted copy of the input array.
    """
    if array.size == 0:
        return array

    return array[jnp.lexsort(array.T[::-1])]

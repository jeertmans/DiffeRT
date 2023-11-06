import jax.numpy as jnp
from jaxtyping import Array, UInt, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def generate_path_candidates(
    num_primitives: int, order: int
) -> UInt[Array, "num_candidates order"]:
    """
    Generate an array of path candidates for fixed path order
    and a number of primitives.

    The returned array contains, for each row, an array of
    ``order`` indices indicating the primitive with which the path interacts.

    This list is generated as the list of all simple paths from one node to
    another, by passing by exactly ``order`` primitives. Calling this function
    is equivalent to calling :func:`itertools.product` with parameters
    ``[0, 1, ..., num_primitives - 1]`` and ``repeat=order``, and removing entries
    with containing loops, i.e., two or more consecutive indices that are equal.

    Args:
        num_primitives: the (positive) number of primitives.
        order: the path order. An order less than one returns an empty array.

    Returns:
        An unsigned array with primitive indices on each row. Its number of rows
        is actually equal to
        ``num_primitives * ((num_primitives - 1) ** (order - 1))``.
    """
    if order < 1:
        return jnp.empty((0, 0), dtype=jnp.uint32)
    elif order == 1:
        indices = jnp.arange(num_primitives, dtype=jnp.uint32)
        return jnp.reshape(indices, (-1, 1))

    num_candidates = num_primitives * ((num_primitives - 1) ** (order - 1))
    path_candidates = jnp.empty((num_candidates, order), dtype=jnp.uint32)
    batch_size = num_candidates // num_primitives

    fill_value = 0
    for j in range(order):
        i = 0
        while i < num_candidates:
            if j > 0 and fill_value == path_candidates[i, j - 1]:
                fill_value = (fill_value + 1) % num_primitives

            path_candidates = path_candidates.at[i : i + batch_size, j].set(fill_value)
            i += batch_size
            fill_value = (fill_value + 1) % num_primitives

        batch_size = batch_size // (num_primitives - 1)

    return path_candidates

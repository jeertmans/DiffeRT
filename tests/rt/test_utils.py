import chex
import jax.numpy as jnp
import pytest
from chex import Array

from differt.rt.utils import generate_path_candidates


def uint_array(array_like: Array) -> Array:
    return jnp.array(array_like, dtype=jnp.uint32)


@pytest.mark.parametrize(
    "num_primitives,order,expected",
    [
        (8, 0, jnp.empty((0, 0), dtype=jnp.uint32)),
        (3, 1, uint_array([[0], [1], [2]])),
        (3, 2, uint_array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])),
        (
            3,
            3,
            uint_array(
                [
                    [0, 1, 0],
                    [0, 1, 2],
                    [0, 2, 0],
                    [0, 2, 1],
                    [1, 0, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [1, 2, 1],
                    [2, 0, 1],
                    [2, 0, 2],
                    [2, 1, 0],
                    [2, 1, 2],
                ]
            ),
        ),
    ],
)
def test_generate_path_candidates(
    num_primitives: int, order: int, expected: Array
) -> None:
    got = generate_path_candidates(num_primitives, order)
    if got.size > 0:
        got = got[jnp.lexsort(got.T[::-1])]
    chex.assert_trees_all_equal_shapes_and_dtypes(got, expected)
    chex.assert_trees_all_equal(got, expected)

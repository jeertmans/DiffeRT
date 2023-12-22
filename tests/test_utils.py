from differt.utils import sorted_array2
import jax.numpy as jnp
import pytest

import chex

from chex import Array

@pytest.mark.parametrize("array,expected", [
    (jnp.empty((0, 0)), jnp.empty((0, 0))),
    (jnp.empty((1, 0)), jnp.empty((1, 0))),
    (jnp.empty((0, 1)), jnp.empty((0, 1))),
    (jnp.ones((4, 5)), jnp.ones((4, 5))),
    (jnp.array([
        [7,8,9],
        [0,1,0],
        [0,0,0],
        [0,0,1],
        [4,5,6],
        [1,2,3],
        [1,0,0],
        [1,0,1],
        ]),
jnp.array([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [1,0,0],
        [1,0,1],
        [1,2,3],
        [4,5,6],
        [7,8,9],
        ]))

    ])
def test_sorted_array2(array: Array, expected: Array) -> None:
    print("input array", array)
    got = sorted_array2(array)

    chex.assert_trees_all_close(got, expected)

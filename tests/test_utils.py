import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from differt.utils import minimize, sorted_array2


@pytest.mark.parametrize(
    "array,expected",
    [
        (jnp.empty((0, 0)), jnp.empty((0, 0))),
        (jnp.empty((1, 0)), jnp.empty((1, 0))),
        (jnp.empty((0, 1)), jnp.empty((0, 1))),
        (jnp.ones((4, 5)), jnp.ones((4, 5))),
        (jnp.arange(9).reshape(3, 3)[::-1, :], jnp.arange(9).reshape(3, 3)),
        (
            jnp.array(
                [
                    [7, 8, 9],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                    [4, 5, 6],
                    [1, 2, 3],
                    [1, 0, 0],
                    [1, 0, 1],
                ]
            ),
            jnp.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ),
        ),
    ],
)
def test_sorted_array2(array: Array, expected: Array) -> None:
    got = sorted_array2(array)

    chex.assert_trees_all_close(got, expected)


def test_minimize() -> None:
    def fun(x: Array, a: Array, b: Array, c: Array) -> Array:
        return (x[..., 0] - a) ** 2.0 + (x[..., 1] - b) ** 2.0 + c

    a = jnp.array([0.0, 1.0, 2.0])
    b = jnp.array([3.0, 4.0, 5.0, 6.0])
    c = jnp.array([7.0, 8.0])

    a, b, c = jnp.meshgrid(a, b, c)
    x0 = jnp.zeros((*a.shape, 2))

    got_x, got_loss = minimize(fun, x0, args=(a, b, c), steps=1000)

    expected_x = jnp.stack((a, b), axis=-1)

    chex.assert_trees_all_close(fun(expected_x, a, b, c), c)

    chex.assert_trees_all_close(got_x, expected_x)
    chex.assert_trees_all_close(got_loss, c)

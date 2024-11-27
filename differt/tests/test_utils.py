import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray

from differt.utils import dot, minimize, sample_points_in_bounding_box, sorted_array2

from .utils import random_inputs


@pytest.mark.parametrize(
    ("u", "v"),
    [
        ((10, 3), (1, 3)),
        ((1, 3), (10, 10, 3)),
    ],
)
@pytest.mark.parametrize("pass_v", [False, True])
@pytest.mark.parametrize("keepdims", [False, True])
@random_inputs("u", "v")
def test_dot(
    u: Array,
    v: Array,
    pass_v: bool,
    keepdims: bool,
) -> None:
    if pass_v:
        got = dot(u, v, keepdims=keepdims)
        expected = jnp.sum(u * v, axis=-1, keepdims=keepdims)
    else:
        got = dot(u, v, keepdims=keepdims)
        expected = jnp.sum(u * v, axis=-1, keepdims=keepdims)

    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    ("array", "expected"),
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
                ],
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
                ],
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

    with pytest.raises(
        TypeError, match="Assertion assert_tree_has_only_ndarrays failed"
    ):
        _ = minimize(fun, x0, args=(0.0, b, c))

    with pytest.raises(TypeError, match="Assertion assert_tree_shape_prefix failed"):
        _ = minimize(fun, x0, args=(a[0, ...], b, c))

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        _ = minimize(fun, x0, args=(a, b))


def test_sample_points_in_bounding_box(key: PRNGKeyArray) -> None:
    def assert_in_bounds(a: Array, bounds: Array) -> None:
        a = a.reshape(-1, a.shape[-1])

        assert bounds.shape[0] == 2
        assert a.shape[1] == bounds.shape[1]

        for i in range(a.shape[1]):
            assert jnp.all(a[:, i] >= bounds[0, i])
            assert jnp.all(a[:, i] <= bounds[1, i])

    bounding_box = jnp.array([[-1.0, -2.0, -3.0], [+4.0, +5.0, +6.0]])

    got = sample_points_in_bounding_box(bounding_box, key=key)

    assert_in_bounds(got, bounding_box)
    assert got.shape == (3,)

    got = sample_points_in_bounding_box(bounding_box, shape=(100,), key=key)

    assert_in_bounds(got, bounding_box)
    assert got.shape == (100, 3)

    got = sample_points_in_bounding_box(bounding_box, shape=(4, 5), key=key)

    assert_in_bounds(got, bounding_box)
    assert got.shape == (4, 5, 3)

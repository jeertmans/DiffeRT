import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from differt.utils import (
    safe_divide,
    sample_points_in_bounding_box,
    smoothing_function,
)


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


def test_safe_divide(key: PRNGKeyArray) -> None:
    key_x, key_y = jax.random.split(key, 2)
    x = jax.random.uniform(key_x, (30, 20))
    y = jax.random.randint(key_y, (30, 20), minval=0, maxval=3)

    assert y.sum() > 0, "We need at least one division by zero"

    got = safe_divide(x, y)

    assert not jnp.all(jnp.isnan(got)), "We don't want any NaN"

    expected = jnp.where(y != 0, x / y, 0)

    chex.assert_trees_all_equal_shapes_and_dtypes(got, expected)
    chex.assert_trees_all_equal(got, expected)


def test_smoothing_function(key: PRNGKeyArray) -> None:
    key_x, key_smoothing_factor = jax.random.split(key, 2)
    x = jax.random.normal(key_x, (40, 1, 10)) * 1000.0
    x = x.at[0, 0, 0].set(0.0)  # Make first entry be a zero
    smoothing_factor = jax.random.uniform(
        key_smoothing_factor, (20, 1), minval=0, maxval=100
    )

    got = smoothing_function(x, smoothing_factor)
    expected = jax.nn.sigmoid(x * smoothing_factor)

    chex.assert_trees_all_equal(got, expected)

    x_limits = jnp.array([-1e8, 0.0, +1e8])

    got = smoothing_function(x_limits, smoothing_factor)
    expected = jnp.broadcast_to(jnp.array([0.0, 0.5, 1.0]), got.shape)

    chex.assert_trees_all_close(got, expected)

    got = smoothing_function(x, 1e8)
    expected = 0.5 * (jnp.sign(x) + 1)
    chex.assert_trees_all_equal(got, expected)

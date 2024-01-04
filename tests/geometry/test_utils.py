from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest
from chex import Array

from differt.geometry.utils import pairwise_cross
from tests.utils import random_inputs


def test_pairwise_cross() -> None:
    u = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cross = pairwise_cross(u, u)
    got = jnp.linalg.norm(cross, axis=-1)
    expected = jnp.ones((3, 3)) - jnp.eye(3, 3)
    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    "u,v,expectation",
    [
        ((10, 3), (10, 3), does_not_raise()),
        ((10, 3), (20, 3), does_not_raise()),
        ((10, 4), (20, 4), pytest.raises(TypeError)),
    ],
)
@random_inputs("u", "v")
def test_pairwise_cross_random_inputs(
    u: Array, v: Array, expectation: AbstractContextManager[Exception]
) -> None:
    with expectation:
        got = pairwise_cross(u, v)

        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                expected = jnp.cross(u[i, :], v[j, :])
                chex.assert_trees_all_equal(got[i, j], expected)

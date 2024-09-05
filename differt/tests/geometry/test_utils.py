from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from differt.geometry.utils import (
    normalize,
    orthogonal_basis,
    pairwise_cross,
    path_lengths,
)
from tests.utils import random_inputs


def test_pairwise_cross() -> None:
    u = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cross = pairwise_cross(u, u)
    got = jnp.linalg.norm(cross, axis=-1)
    expected = jnp.ones((3, 3)) - jnp.eye(3, 3)
    chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize(
    ("u", "v", "expectation"),
    [
        ((10, 3), (10, 3), does_not_raise()),
        ((10, 3), (20, 3), does_not_raise()),
        ((10, 4), (20, 4), pytest.raises(TypeError)),
    ],
)
@random_inputs("u", "v")
def test_pairwise_cross_random_inputs(
    u: Array,
    v: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = pairwise_cross(u, v)

        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                expected = jnp.cross(u[i, :], v[j, :])
                chex.assert_trees_all_close(got[i, j], expected, atol=1e-7)


@pytest.mark.parametrize(
    ("u", "expectation"),
    [
        ((10, 3), does_not_raise()),
        ((20, 10, 3), does_not_raise()),
        ((10, 4), pytest.raises(TypeError)),
    ],
)
@random_inputs("u")
def test_normalize_random_inputs(
    u: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        nu, lu = normalize(u)

        chex.assert_trees_all_close(u, nu * lu[..., None])


@pytest.mark.parametrize(
    "u",
    [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
        jnp.array([1.0, 1.0, 1.0]),
        jnp.arange(30.0).reshape(2, 5, 3),
    ],
)
def test_orthogonal_basis(u: Array) -> None:
    u, _ = normalize(u)
    v, w = orthogonal_basis(u)

    for vec in [v, w]:
        # Vectors should be perpendicular
        dot = jnp.sum(u * vec, axis=-1)
        chex.assert_trees_all_close(dot, 0.0, atol=1e-7)
        # Vectors should have unit length
        _, length = normalize(vec)
        chex.assert_trees_all_close(length, 1.0)

    # Vectors should be perpendicular
    dot = jnp.sum(u * v, axis=-1)
    chex.assert_trees_all_close(dot, 0.0, atol=1e-7)


@pytest.mark.parametrize(
    ("paths", "expectation"),
    [
        ((10, 3), does_not_raise()),
        ((20, 10, 3), does_not_raise()),
        ((10, 4), pytest.raises(TypeError)),
        ((1, 3), does_not_raise()),
        ((0, 3), does_not_raise()),
    ],
)
@random_inputs("paths")
def test_path_lengths_random_inputs(
    paths: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = path_lengths(paths)
        expected = jnp.sum(jnp.linalg.norm(jnp.diff(paths, axis=-2), axis=-1), axis=-1)

        chex.assert_trees_all_close(got, expected)

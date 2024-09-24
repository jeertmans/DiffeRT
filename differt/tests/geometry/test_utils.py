from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, DTypeLike, Float, PRNGKeyArray

from differt.geometry.utils import (
    assemble_paths,
    fibonacci_lattice,
    normalize,
    orthogonal_basis,
    pairwise_cross,
    path_lengths,
    rotation_matrix_along_axis,
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
)

from ..utils import random_inputs


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


@pytest.mark.parametrize("sign", [+1.0, -1.0])
@pytest.mark.parametrize("flip_axis", [False, True])
@pytest.mark.parametrize(
    ("axis", "func"),
    [
        ((1.0, 0.0, 0.0), rotation_matrix_along_x_axis),
        ((0.0, 1.0, 0.0), rotation_matrix_along_y_axis),
        ((0.0, 0.0, 1.0), rotation_matrix_along_z_axis),
    ],
)
def test_rotation_matrices(
    sign: float,
    flip_axis: bool,
    axis: tuple[float, float, float],
    func: Callable[[Float[ArrayLike, " "]], Float[Array, "3 3"]],
    key: PRNGKeyArray,
) -> None:
    angle = jax.random.uniform(key, minval=0.0, maxval=2.0 * jnp.pi)
    angle = jnp.copysign(angle, sign)

    if flip_axis:
        expected = func(-angle)
        got = rotation_matrix_along_axis(angle, -jnp.array(axis))
    else:
        expected = func(+angle)
        got = rotation_matrix_along_axis(angle, +jnp.array(axis))

    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize("n", [0, 10, 100])
@pytest.mark.parametrize(
    ("dtype", "expected_dtype", "expectation"),
    [
        (None, jnp.float32, does_not_raise()),
        (jnp.float32, jnp.float32, does_not_raise()),
        (float, jnp.float64, does_not_raise()),
        ("float", jnp.float64, does_not_raise()),
        (jnp.float16, jnp.float16, does_not_raise()),
        (jnp.float64, jnp.float64, does_not_raise()),
        (
            int,
            jnp.float32,
            pytest.raises(
                ValueError,
                match="Unsupported dtype <class 'int'>, must be a floating dtype.",
            ),
        ),
        (
            jnp.int32,
            jnp.float32,
            pytest.raises(
                ValueError,
                match="Unsupported dtype <class 'jax.numpy.int32'>, must be a floating dtype.",
            ),
        ),
    ],
)
def test_fibonacci_lattice(
    n: int,
    dtype: DTypeLike | None,
    expected_dtype: jnp.dtype,
    expectation: AbstractContextManager[Exception],
) -> None:
    with jax.experimental.enable_x64(expected_dtype == jnp.float64), expectation:  # type: ignore[reportAttributeAccessIssue]
        got = fibonacci_lattice(n, dtype=dtype)

        normalized, lengths = normalize(got)

        atol = jnp.finfo(expected_dtype).eps

        chex.assert_type(got, expected_dtype)
        chex.assert_trees_all_close(got, normalized, atol=atol)
        chex.assert_trees_all_close(lengths, jnp.ones_like(lengths), atol=atol)


@pytest.mark.parametrize(
    ("shapes", "expectation"),
    [
        (((2, 3),), does_not_raise()),
        (((1, 3), (2, 3), (1, 3)), does_not_raise()),
        (((1, 3), (10, 5, 2, 3), (1, 3)), does_not_raise()),
        (((1, 3), (2, 4), (1, 3)), pytest.raises(TypeError)),
        (((1, 3), (3,), (1, 3)), pytest.raises(TypeError)),
        (((10, 1, 3), (10, 2, 3), (5, 1, 3)), pytest.raises(TypeError)),
    ],
)
def test_assemble_paths(
    shapes: tuple[tuple[int, ...], ...],
    expectation: AbstractContextManager[Exception],
    key: PRNGKeyArray,
) -> None:
    keys = jax.random.split(key, len(shapes))

    path_segments = [
        jax.random.uniform(key, shape=shape)
        for key, shape in zip(keys, shapes, strict=False)
    ]

    with expectation:
        got = assemble_paths(*path_segments)
        expected_path_length = sum(shape[-2] for shape in shapes)

        assert got.shape[-2] == expected_path_length

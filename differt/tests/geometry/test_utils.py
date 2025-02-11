from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from itertools import product

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, ArrayLike, DTypeLike, Float, PRNGKeyArray

from differt.geometry._utils import (
    assemble_paths,
    cartesian_to_spherical,
    fibonacci_lattice,
    min_distance_between_cells,
    normalize,
    orthogonal_basis,
    pairwise_cross,
    path_lengths,
    perpendicular_vectors,
    rotation_matrix_along_axis,
    rotation_matrix_along_x_axis,
    rotation_matrix_along_y_axis,
    rotation_matrix_along_z_axis,
    spherical_to_cartesian,
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
@pytest.mark.parametrize("keepdims", [False, True])
@random_inputs("u")
def test_normalize_random_inputs(
    u: Array,
    keepdims: bool,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        nu, lu = normalize(u, keepdims=keepdims)

        if keepdims:
            chex.assert_trees_all_close(u, nu * lu)
        else:
            chex.assert_trees_all_close(u, nu * lu[..., None])


def test_perpendicular_vectors() -> None:
    all_vectors = list(product([0.0, 1.0, -1.0], repeat=3))
    # Drop [0, 0, 0] case
    all_vectors = all_vectors[1:]
    u = jnp.array(all_vectors)
    v = perpendicular_vectors(u)

    chex.assert_trees_all_close(jnp.linalg.norm(v, axis=-1), 1.0)
    chex.assert_trees_all_close(jnp.sum(u * v, axis=-1), 0.0)


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
                match=r"Unsupported dtype <class 'int'>, must be a floating dtype.",
            ),
        ),
        (
            jnp.int32,
            jnp.float32,
            pytest.raises(
                ValueError,
                match=r"Unsupported dtype <class 'jax.numpy.int32'>, must be a floating dtype.",
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


def test_assemble_paths() -> None:
    tx = jnp.ones((5, 3))
    paths = jnp.arange(24.0).reshape(4, 2, 3)
    rx = -jnp.ones((6, 3))

    expected = jnp.concatenate(
        (
            jnp.tile(tx[:, None, None, None, :], (1, 6, 4, 1, 1)),
            jnp.tile(paths[None, None, ...], (5, 6, 1, 1, 1)),
            jnp.tile(rx[None, :, None, None, :], (5, 1, 4, 1, 1)),
        ),
        axis=-2,
    )

    got = assemble_paths(tx[:, None, None, None, :], paths, rx[None, :, None, None, :])

    chex.assert_trees_all_equal(got, expected)


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
def test_assemble_paths_random_inputs(
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


def test_cartesian_to_spherial_roundtrip(key: PRNGKeyArray) -> None:
    key_xyz, key_sph = jax.random.split(key, 2)

    xyz = 10 * jax.random.normal(key_xyz, (100, 3))
    sph = cartesian_to_spherical(xyz)
    got = spherical_to_cartesian(sph)

    chex.assert_trees_all_close(got, xyz, atol=2e-5)

    key_r, key_polar, key_azim = jax.random.split(key_sph, 3)

    r = jnp.abs(10 * jax.random.normal(key_r, (100,)))
    p = jax.random.uniform(key_polar, (100,), minval=0, maxval=jnp.pi)
    a = jax.random.uniform(key_azim, (100,), minval=-jnp.pi, maxval=jnp.pi)
    sph = jnp.stack((r, p, a), axis=-1)
    xyz = spherical_to_cartesian(sph)
    got = cartesian_to_spherical(xyz)

    chex.assert_trees_all_close(got, sph, atol=7e-5)


def test_min_distance_between_cells(key: PRNGKeyArray) -> None:
    key_vertices, key_ids = jax.random.split(key, 2)

    batch = (10, 4, 5)

    cell_vertices = 10 * jax.random.normal(key_vertices, (*batch, 3))
    cell_ids = jax.random.randint(key_ids, batch, minval=0, maxval=5)

    got = min_distance_between_cells(cell_vertices, cell_ids).reshape(-1)

    cell_vertices = cell_vertices.reshape(-1, 3)
    cell_ids = cell_ids.reshape(-1)

    for cell_vertex, cell_id, got_dist in zip(
        cell_vertices, cell_ids, got, strict=False
    ):
        dist = jnp.linalg.norm(cell_vertex - cell_vertices, axis=-1)
        expected_dist = jnp.min(dist, where=(cell_id != cell_ids), initial=jnp.inf)

        chex.assert_trees_all_close(got_dist, expected_dist)

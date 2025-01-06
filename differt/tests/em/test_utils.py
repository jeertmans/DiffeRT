from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax
import jax.experimental
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray

from differt.em._constants import c
from differt.em._utils import (
    lengths_to_delays,
    path_delays,
    sp_directions,
    sp_rotation_matrix,
)
from differt.geometry import normalize, rotation_matrix_along_z_axis
from differt.utils import dot

from ..utils import random_inputs


@pytest.mark.parametrize(
    ("lengths", "speed", "expectation"),
    [
        ((10,), (1,), does_not_raise()),
        ((10,), (2,), pytest.raises(TypeError)),
        ((20, 10), (1,), does_not_raise()),
        ((20, 10), (10,), does_not_raise()),
        ((20, 1), (10,), does_not_raise()),
        ((20, 1), (1, 10), does_not_raise()),
        ((20, 1), (), does_not_raise()),
        ((20, 10), (20,), pytest.raises(TypeError)),
        ((10, 4), (10, 5), pytest.raises(TypeError)),
    ],
)
@random_inputs("lengths", "speed")
def test_lengths_to__delays_random_inputs(
    lengths: Array,
    speed: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = lengths_to_delays(lengths, speed=speed)
        expected = lengths / speed

        chex.assert_trees_all_close(got, expected)


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
def test_path_delays_random_inputs(
    paths: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = path_delays(paths)
        expected = (
            jnp.sum(jnp.linalg.norm(jnp.diff(paths, axis=-2), axis=-1), axis=-1) / c
        )

        chex.assert_trees_all_close(got, expected)


def test_sp_directions() -> None:
    cos = jnp.cos(jnp.pi / 6)
    sin = jnp.sin(jnp.pi / 6)
    k_i = jnp.array([[cos, -sin, 0.0], [0.0, -1.0, 0.0]])
    k_r = jnp.array([[cos, +sin, 0.0], [0.0, +1.0, 0.0]])
    normals = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    got = sp_directions(k_i, k_r, normals)

    chex.assert_trees_all_close(
        got[0][0], got[1][0], custom_message="s-components should be equal"
    )

    for comp, k in zip(got, (k_i, k_r), strict=True):
        s = comp[0]
        p = comp[1]

        chex.assert_trees_all_close(jnp.cross(p, s), k)
        chex.assert_trees_all_close(jnp.cross(k, p), s)
        chex.assert_trees_all_close(jnp.cross(s, k), p)

    expected_e_i_s = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    expected_e_i_p = jnp.array([[+sin, cos, 0.0], [0.0, 0.0, -1.0]])
    expected_e_r_p = jnp.array([[-sin, cos, 0.0], [0.0, 0.0, 1.0]])

    chex.assert_trees_all_close(got[0][0], expected_e_i_s)
    chex.assert_trees_all_close(got[0][1], expected_e_i_p)
    chex.assert_trees_all_close(got[1][1], expected_e_r_p)


def test_sp_rotation_matrix() -> None:
    e_i_s = jnp.array([1.0, 0.0, 0.0])
    e_i_p = jnp.array([0.0, 1.0, 0.0])

    e_r_s = jnp.array([+0.0, 1.0, 0.0])
    e_r_p = jnp.array([-1.0, 0.0, 0.0])

    got_R = sp_rotation_matrix(e_i_s, e_i_p, e_r_s, e_r_p)
    expected_R = rotation_matrix_along_z_axis(-jnp.pi / 2)

    chex.assert_trees_all_close(got_R, expected_R[:-1, :-1], atol=1e-7)
    chex.assert_trees_all_close(got_R @ got_R.mT, jnp.eye(2))

    e_r_s = jnp.array([+1.0, 1.0, 0.0]) * jnp.sqrt(2) / 2
    e_r_p = jnp.array([-1.0, 1.0, 0.0]) * jnp.sqrt(2) / 2

    got_R = sp_rotation_matrix(e_i_s, e_i_p, e_r_s, e_r_p)
    expected_R = rotation_matrix_along_z_axis(-jnp.pi / 4)

    chex.assert_trees_all_close(got_R, expected_R[:-1, :-1])
    chex.assert_trees_all_close(got_R @ got_R.mT, jnp.eye(2), atol=1e-7)

    e_r_s = +e_i_s
    e_r_p = -e_i_p

    got_R = sp_rotation_matrix(e_i_s, e_i_p, e_r_s, e_r_p)
    expected_R = rotation_matrix_along_z_axis(0.0).at[1, 1].set(-1.0)

    # Improper rotation matrix, determinant should be -1.0
    chex.assert_trees_all_close(jnp.linalg.det(got_R), -1.0)
    chex.assert_trees_all_close(got_R, expected_R[:-1, :-1])
    chex.assert_trees_all_close(got_R @ got_R.mT, jnp.eye(2))


@pytest.mark.parametrize(
    "batch",
    [
        # (),
        (4,),
        # (10,),
        # (20, 10),
    ],
)
def test_sp_rotation_matrix_random_inputs(
    batch: tuple[int, ...], key: PRNGKeyArray
) -> None:
    # TODO: replace with jax.numpy.matvec once it becomes available
    def matvec(R: Array, v: Array) -> Array:
        return jnp.einsum("...ij,...j->...i", R, v)

    key_E, key_e = jax.random.split(key, 2)
    key_E_real, key_E_imag = jax.random.split(key_E, 2)
    E_i = jax.random.normal(key_E_real, (*batch, 3)) + jax.random.normal(
        key_E_imag, (*batch, 3)
    )

    # Generate sound directions for the incident and reflected waves

    k_i, normals = jnp.unstack(
        normalize(jax.random.normal(key_e, (2, *batch, 3)).at[..., -1].set(0.0))[0],
        axis=0,
    )
    chex.assert_trees_all_close(jnp.linalg.norm(k_i, axis=-1), 1.0)

    # Ensure the normals are pointing towards the incident wave

    normals = jnp.where(dot(normals, k_i, keepdims=True) >= 0.0, -normals, normals)
    chex.assert_trees_all_close(jnp.linalg.norm(normals, axis=-1), 1.0)

    # Generate reflected wave direction

    k_r = k_i - 2.0 * dot(k_i, normals, keepdims=True) * normals
    chex.assert_trees_all_close(jnp.linalg.norm(k_r, axis=-1), 1.0)
    chex.assert_trees_all_equal(dot(k_i, k_r) <= 0.0, True)

    print(f"{k_i=}, {normals=}, {k_r=}")

    # Generate s and p components for the incident and reflected waves

    (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, normals)

    print(f"{e_i_s=}, {e_i_p=}, {e_r_s=}, {e_r_p=}")

    # Generated s and p components should be orthogonal

    chex.assert_trees_all_close(dot(e_i_s, e_i_p), 0.0, atol=1e-7)
    chex.assert_trees_all_close(dot(e_i_s, k_i), 0.0, atol=1e-7)
    chex.assert_trees_all_close(dot(e_i_p, k_i), 0.0, atol=1e-7)
    chex.assert_trees_all_close(dot(e_r_s, e_r_p), 0.0, atol=1e-7)
    chex.assert_trees_all_close(dot(e_r_s, k_r), 0.0, atol=1e-7)
    chex.assert_trees_all_close(dot(e_r_p, k_r), 0.0, atol=1e-7)

    # The s and p components should be normalized

    chex.assert_trees_all_close(jnp.linalg.norm(e_i_s, axis=-1), 1.0)
    chex.assert_trees_all_close(jnp.linalg.norm(e_i_p, axis=-1), 1.0)
    chex.assert_trees_all_close(jnp.linalg.norm(e_r_s, axis=-1), 1.0)
    chex.assert_trees_all_close(jnp.linalg.norm(e_r_p, axis=-1), 1.0)

    E_i_s_p = jnp.concatenate(
        (dot(E_i, e_i_s, keepdims=True), dot(E_i, e_i_p, keepdims=True)), axis=-1
    )

    # Remove E components orthogonal to the s and p components

    E_i = E_i_s_p[..., 0, None] * e_i_s + E_i_s_p[..., 1, None] * e_i_p

    # Generate the rotation matrix

    R = sp_rotation_matrix(e_i_s, e_i_p, e_r_s, e_r_p)

    print(f"{R=}")

    # chex.assert_trees_all_close(
    #    jax.numpy.linalg.det(R),
    #    -1.0,
    # )

    # The rotation matrix should be orthogonal

    chex.assert_trees_all_close(
        R @ R.mT,
        jnp.broadcast_to(jnp.eye(2), R.shape),
    )

    E_r_s_p = matvec(R, E_i_s_p)

    E_r = E_r_s_p[..., 0, None] * e_r_s + E_r_s_p[..., 1, None] * e_r_p

    chex.assert_trees_all_equal_shapes_and_dtypes(
        E_r,
        E_i,
    )

    # The norm should be preserved.

    # chex.assert_trees_all_close(
    #    jnp.linalg.norm(E_r, axis=-1),
    #    jnp.linalg.norm(E_i, axis=-1),
    # )

    # A back-rotation should yield the original field

    chex.assert_trees_all_close(
        matvec(R.mT, E_r_s_p),
        E_i_s_p,
    )

# ruff: noqa: N806
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray

from differt.em import Dipole, c
from differt.em._utils import (
    fspl,
    lengths_to_delays,
    path_delays,
    sp_directions,
    sp_rotation_matrix,
)
from differt.geometry import rotation_matrix_along_z_axis, spherical_to_cartesian

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

    # We test normal incidence
    e_r_s = +e_i_s
    e_r_p = -e_i_p

    got_R = sp_rotation_matrix(e_i_s, e_i_p, e_r_s, e_r_p)
    expected_R = rotation_matrix_along_z_axis(0.0).at[1, 1].set(-1.0)

    # Improper rotation matrix, determinant should be -1.0
    chex.assert_trees_all_close(jnp.linalg.det(got_R), -1.0)
    chex.assert_trees_all_close(got_R, expected_R[:-1, :-1])
    chex.assert_trees_all_close(got_R @ got_R.mT, jnp.eye(2))


def test_fspl(key: PRNGKeyArray) -> None:
    key_d, key_f = jax.random.split(key, 2)
    d = jax.random.uniform(key_d, (30, 1), minval=1.0, maxval=100.0)
    f = jax.random.uniform(key_f, (1, 50), minval=0.1e9, maxval=10e9)

    got = fspl(d, f)
    got_db = fspl(d, f, dB=True)
    expected_db = 20 * jnp.log10(d) + 20 * jnp.log10(f) - 147.55

    chex.assert_trees_all_close(10 * jnp.log10(got), got_db)
    chex.assert_trees_all_close(got_db, expected_db, rtol=2e-4)


@pytest.mark.parametrize("frequency", [0.1e9, 1e9, 10e9])
def test_fspl_vs_los(frequency: float, key: PRNGKeyArray) -> None:
    key_r, key_azim, key_current = jax.random.split(key, 3)
    r = jax.random.uniform(key_r, (1000,), minval=10.0, maxval=1000.0)
    azim = jax.random.uniform(key_azim, (1000,), maxval=2 * jnp.pi)
    polar = jnp.full_like(
        azim, jnp.pi / 2
    )  # 90 degrees, direction of maximum radiation
    rpa = jnp.stack([r, polar, azim], axis=-1)
    xyz = spherical_to_cartesian(rpa)
    d = r
    ant = Dipole(
        frequency=frequency,
        current=jax.random.uniform(key_current, minval=1.0, maxval=10.0),
    )

    got = 10 * jnp.log10(
        ant.aperture
        * jnp.linalg.norm(ant.pointing_vector(xyz), axis=-1)
        / ant.reference_power
    )
    expected = -fspl(d, frequency, dB=True)

    chex.assert_trees_all_close(got, expected, rtol=2e-4)

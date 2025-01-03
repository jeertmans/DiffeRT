from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax
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
from differt.geometry import normalize
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


@pytest.mark.parametrize(
    "batch",
    [
        (4,),
        (10,),
        (20, 10),
    ],
)
def test_sp_rotation_matrix_random_inputs(
    batch: tuple[int, ...], key: PRNGKeyArray
) -> None:
    key_E, key_e = jax.random.split(key, 2)
    E_i = jax.random.normal(key_E, (*batch, 3))
    k_i, k_r, normals = jnp.unstack(
        normalize(jax.random.normal(key_e, (3, *batch, 3)))[0], axis=0
    )
    (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, normals)

    # Generated s and p components should be orthogonal

    chex.assert_trees_all_close(dot(e_i_s, e_i_p), 0.0, atol=1e-7)
    chex.assert_trees_all_close(dot(e_r_s, e_r_p), 0.0, atol=1e-7)

    E_i_s_p = jnp.concatenate(
        (dot(E_i, e_i_s, keepdims=True), dot(E_i, e_i_p, keepdims=True)), axis=-1
    )

    # Remove E components orthogonal to the s and p components

    E_i = E_i_s_p[..., 0, None] * e_i_s + E_i_s_p[..., 1, None] * e_i_p

    R = sp_rotation_matrix(e_i_s, e_i_p, e_r_s, e_r_p)

    E_r_s_p = jnp.einsum("...ri,...i->...r", R, E_i_s_p)

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

    R = jnp.moveaxis(R, -1, -2)

    chex.assert_trees_all_close(
        jnp.einsum("...ir,...r->...i", R, E_r_s_p),
        E_i_s_p,
    )

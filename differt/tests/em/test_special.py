from contextlib import contextmanager

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp
from chex import Array

from differt.em.special import erf, erfc, fresnel


@contextmanager
def enable_double_precision(enable: bool):
    enabled = jax.config.jax_enable_x64  # type: ignore[attr-defined]
    try:
        jax.config.update("jax_enable_x64", enable)
        yield
    finally:
        jax.config.update("jax_enable_x64", enabled)


@pytest.mark.parametrize(
    "double_precision",
    (False, True),
)
def test_erf(double_precision: bool) -> None:
    with enable_double_precision(double_precision):
        t = jnp.linspace(-6.0, 6.0, 101)
        a, b = jnp.meshgrid(t, t)
        z = a + 1j * b
        z = z.astype(dtype=jnp.complex128 if double_precision else jnp.complex64)
        got = erf(z)
        expected = jnp.asarray(sp.erf(np.asarray(z)))
        chex.assert_trees_all_close(
            got, expected, rtol=1e-12 if double_precision else 1e-4
        )


@pytest.mark.parametrize(
    "z",
    (
        jnp.linspace(-5.0, 5.0, 101),
        1j * jnp.linspace(-5.0, 5.0, 101),
    ),
)
def test_erfc(z: Array) -> None:
    got = erfc(z)
    expected = jnp.asarray(erfc(np.asarray(z)))
    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    "z",
    (
        jnp.linspace(-5.0, 5.0, 101),
        1j * jnp.linspace(-5.0, 5.0, 101),
    ),
)
def test_fresnel(z: Array) -> None:
    got_s, got_c = fresnel(z)
    expected = fresnel(np.asarray(z))
    expected_s = jnp.asarray(expected[0])
    expected_c = jnp.asarray(expected[1])
    chex.assert_trees_all_close(got_s, expected_s)
    chex.assert_trees_all_close(got_c, expected_c)


def test_fresnel_special_real_arguments() -> None:
    s, c = fresnel(0.0)
    chex.assert_trees_all_close(s, 0.0, atol=1e-6)
    chex.assert_trees_all_close(c, 0.0, atol=1e-6)

    s, c = fresnel(1.0)
    chex.assert_trees_all_close(s, 0.4382591473903547, atol=1e-6)
    chex.assert_trees_all_close(c, 0.779893400376823, atol=1e-6)

    s, c = fresnel(10.0)
    chex.assert_trees_all_close(s, 0.46816997858488224, atol=1e-2)
    chex.assert_trees_all_close(c, 0.49989869420551575, atol=1e-2)

    s_neg, c_neg = fresnel(-1.0)
    s_pos, c_pos = fresnel(+1.0)
    chex.assert_trees_all_close(s_neg, -s_pos, atol=1e-6)
    chex.assert_trees_all_close(c_neg, -c_pos, atol=1e-6)

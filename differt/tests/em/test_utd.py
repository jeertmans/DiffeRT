# pyright: reportMissingTypeArgument=false
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp
from jaxtyping import PRNGKeyArray

from differt.em._utd import F, L_i, diffraction_coefficients


def test_L_i(key: PRNGKeyArray) -> None:  # noqa: N802
    key_s_d, key_sin, key_1_i, key_2_i, key_e_i, key_s_i = jax.random.split(key, 6)

    s_d = jax.random.uniform(key_s_d, (100,), minval=10.0, maxval=100.0)
    sin_2_beta_0 = jax.random.uniform(key_sin, (100,), minval=0.0, maxval=1.0)
    rho_1_i = jax.random.uniform(key_1_i, (100,), minval=10.0, maxval=100.0)
    rho_2_i = jax.random.uniform(key_2_i, (100,), minval=10.0, maxval=100.0)
    rho_e_i = jax.random.uniform(key_e_i, (100,), minval=10.0, maxval=100.0)
    s_i = jax.random.uniform(key_s_i, (100,), minval=10.0, maxval=100.0)

    got = L_i(s_d, sin_2_beta_0)
    expected = s_d * sin_2_beta_0

    chex.assert_trees_all_close(got, expected)

    got = L_i(s_d, sin_2_beta_0, s_i=s_i)
    expected = L_i(s_d, sin_2_beta_0, rho_1_i=s_i, rho_2_i=s_i, rho_e_i=s_i)

    chex.assert_trees_all_close(got, expected)

    got = L_i(s_d, sin_2_beta_0, rho_1_i=rho_1_i, rho_2_i=rho_2_i, rho_e_i=rho_e_i)
    expected = (
        s_d
        * (rho_e_i + s_d)
        * rho_1_i
        * rho_2_i
        * sin_2_beta_0
        / (rho_e_i * (rho_1_i + s_d) * (rho_2_i + s_d))
    )

    chex.assert_trees_all_close(got, expected)

    with pytest.raises(
        ValueError,
        match="If 's_i' is provided, then 'rho_1_i', 'rho_2_i', and 'rho_e_i' must be left to 'None'",
    ):
        _ = L_i(
            s_d,
            sin_2_beta_0,
            rho_1_i=rho_1_i,
            rho_2_i=rho_2_i,
            rho_e_i=rho_e_i,
            s_i=s_i,
        )

    with pytest.raises(
        ValueError,
        match="If 's_i' is provided, then 'rho_1_i', 'rho_2_i', and 'rho_e_i' must be left to 'None'",
    ):
        _ = L_i(s_d, sin_2_beta_0, rho_1_i=rho_1_i, s_i=s_i)

    with pytest.raises(
        ValueError,
        match="All three of 'rho_1_i', 'rho_2_i', and 'rho_e_i' must be provided, or left to 'None'",
    ):
        _ = L_i(s_d, sin_2_beta_0, rho_1_i=rho_1_i)


def scipy_F(x: np.ndarray) -> np.ndarray:  # noqa: N802
    factor = np.sqrt(np.pi / 2)
    sqrtx = np.sqrt(x)

    S, C = sp.fresnel(sqrtx / factor)  # noqa: N806

    return 2j * sqrtx * np.exp(1j * x) * (factor * ((1 - 1j) / 2 - C + 1j * S))


def test_F() -> None:  # noqa: N802
    # Test case 1: 0.001 to 10.0
    x = jnp.logspace(-3, 1, 1000)
    got = F(x)
    expected = jnp.asarray(scipy_F(np.asarray(x)))

    chex.assert_trees_all_close(got, expected, rtol=1e-5)

    # Test case 2: F(x), x -> 0
    info = jnp.finfo(float)
    got = F(info.eps)
    mag = jnp.abs(got)
    angle = jnp.angle(got, deg=True)

    chex.assert_trees_all_close(mag, 0.0, atol=1e-7)
    chex.assert_trees_all_close(angle, 45)

    # Test case 3: F(x), x -> +oo
    got = F(1e6)
    mag = jnp.abs(got)

    chex.assert_trees_all_close(mag, 1.0, atol=1e-4)


def test_diffraction_coefficients() -> None:
    with pytest.raises(NotImplementedError):
        _ = diffraction_coefficients()

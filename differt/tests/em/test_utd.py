# pyright: reportMissingTypeArgument=false
# ruff: noqa: N802, N806, ANN001, ANN202
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp
from jaxtyping import PRNGKeyArray

from differt.em._utd import F, L_i, diffraction_coefficients


def test_L_i(key: PRNGKeyArray) -> None:
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


def scipy_F(x: np.ndarray) -> np.ndarray:
    factor = np.sqrt(np.pi / 2)
    sqrtx = np.sqrt(x)

    S, C = sp.fresnel(sqrtx / factor)

    return 2j * sqrtx * np.exp(1j * x) * (factor * ((1 - 1j) / 2 - C + 1j * S))


def test_F() -> None:
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
    # Scalar inputs
    wavenumber = 50.0
    n = 1.5
    phi_i = 0.5
    phi_d = 1.0
    L_i = 10.0

    D_s, D_h = diffraction_coefficients(wavenumber, n, phi_i, phi_d, L_i)
    assert D_s.shape == ()
    assert D_h.shape == ()

    # Batched inputs
    wavenumber_batch = jnp.array([50.0, 60.0])
    n_batch = jnp.array([1.5, 1.5])
    phi_i_batch = jnp.array([0.5, 0.6])
    phi_d_batch = jnp.array([1.0, 1.1])
    L_i_batch = jnp.array([10.0, 12.0])

    D_s_batch, D_h_batch = diffraction_coefficients(
        wavenumber_batch, n_batch, phi_i_batch, phi_d_batch, L_i_batch
    )
    assert D_s_batch.shape == (2,)
    assert D_h_batch.shape == (2,)

    # Test JIT compilation
    jit_diffraction = jax.jit(diffraction_coefficients)
    D_s_jit, D_h_jit = jit_diffraction(wavenumber, n, phi_i, phi_d, L_i)
    chex.assert_trees_all_close(D_s, D_s_jit)
    chex.assert_trees_all_close(D_h, D_h_jit)

    # Test gradients with respect to phi_i
    def sum_diffraction(phi):
        D_s, D_h = diffraction_coefficients(wavenumber, n, phi, phi_d, L_i)
        return jnp.real(D_s) + jnp.real(D_h)

    grad_fn = jax.grad(sum_diffraction)
    grad_val = grad_fn(phi_i)
    assert not jnp.isnan(grad_val)

    # Test continuity across shadow boundaries (where phi_d - phi_i = pi)
    phi_boundary = phi_i + jnp.pi
    eps = 1e-6
    D_s_left, D_h_left = diffraction_coefficients(
        wavenumber, n, phi_i, phi_boundary - eps, L_i
    )
    D_s_right, D_h_right = diffraction_coefficients(
        wavenumber, n, phi_i, phi_boundary + eps, L_i
    )

    # Since they are on the shadow boundary, they should be well-behaved and finite (no NaNs)
    assert not jnp.any(jnp.isnan(D_s_left))
    assert not jnp.any(jnp.isnan(D_h_left))
    assert not jnp.any(jnp.isnan(D_s_right))
    assert not jnp.any(jnp.isnan(D_h_right))

    # Test exact boundary evaluation
    D_s_exact, D_h_exact = diffraction_coefficients(
        wavenumber, n, phi_i, phi_boundary, L_i
    )
    assert not jnp.any(jnp.isnan(D_s_exact))
    assert not jnp.any(jnp.isnan(D_h_exact))


def test_lossy_diffraction_coefficients() -> None:
    # Scalar inputs
    wavenumber = 50.0
    n = 1.5
    phi_i = 0.5
    phi_d = 1.0
    L_i = 10.0

    # 1. Verify default parameters behave like PEC (None)
    D_s_def, D_h_def = diffraction_coefficients(wavenumber, n, phi_i, phi_d, L_i)
    D_s_pec, D_h_pec = diffraction_coefficients(
        wavenumber, n, phi_i, phi_d, L_i, n_r_o=None, n_r_n=None
    )
    chex.assert_trees_all_close(D_s_def, D_s_pec)
    chex.assert_trees_all_close(D_h_def, D_h_pec)

    # 2. Verify large refractive index (lossy material approaching PEC)
    # Note: As relative refractive index n_r -> infinity, reflection coefficients
    # r_s -> -1 and r_p -> +1, so the result should approach the PEC case.
    large_n_r = 1e6 + 0j
    D_s_lossy, D_h_lossy = diffraction_coefficients(
        wavenumber, n, phi_i, phi_d, L_i, n_r_o=large_n_r, n_r_n=large_n_r
    )
    chex.assert_trees_all_close(D_s_lossy, D_s_pec, rtol=1e-4, atol=1e-4)
    chex.assert_trees_all_close(D_h_lossy, D_h_pec, rtol=1e-4, atol=1e-4)

    # 3. Verify JIT compilation with lossy material parameters
    jit_diff = jax.jit(diffraction_coefficients)
    n_r_o = 3.0 - 0.5j
    n_r_n = 4.0 - 1.0j
    D_s_jit, D_h_jit = jit_diff(
        wavenumber, n, phi_i, phi_d, L_i, n_r_o=n_r_o, n_r_n=n_r_n
    )
    assert not jnp.any(jnp.isnan(D_s_jit))
    assert not jnp.any(jnp.isnan(D_h_jit))

    # 4. Verify gradients with respect to phi_i with lossy materials
    def sum_lossy_diff(phi):
        D_s, D_h = diffraction_coefficients(
            wavenumber, n, phi, phi_d, L_i, n_r_o=n_r_o, n_r_n=n_r_n
        )
        return jnp.real(D_s) + jnp.real(D_h)

    grad_fn = jax.grad(sum_lossy_diff)
    grad_val = grad_fn(phi_i)
    assert not jnp.isnan(grad_val)

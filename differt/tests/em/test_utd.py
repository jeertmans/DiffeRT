import chex
import jax.numpy as jnp
import numpy as np
import scipy.special as sp

from differt.em.utd import F, diffraction_coefficients


def scipy_F(x: np.ndarray) -> np.ndarray:  # noqa: N802
    factor = np.sqrt(np.pi / 2)
    sqrtx = np.sqrt(x)

    S, C = sp.fresnel(sqrtx / factor)  # noqa: N806

    return 2j * sqrtx * np.exp(1j * x) * (factor * ((1 - 1j) / 2 - C + 1j * S))


def test_F() -> None:  # noqa: N802
    x = jnp.logspace(-3, 1, 100)
    got = F(x)
    expected = jnp.asarray(scipy_F(np.asarray(x)))

    chex.assert_trees_all_close(got, expected, rtol=1e-5)


def test_diffraction_coefficients():
    # Test case 1: Normal diffraction
    incident_ray = jnp.array([1.0, 0.0, 0.0])
    diffracted_ray = jnp.array([0.0, 1.0, 0.0])
    edge_vector = jnp.array([0.0, 0.0, 1.0])
    k = 2 * jnp.pi  # Assuming wavelength = 1
    n = 1.5
    r_prime = 10.0
    r = 20.0
    r0 = 5.0
    result = diffraction_coefficients(
        incident_ray, diffracted_ray, edge_vector, k, n, r_prime, r, r0
    )
    assert result.shape == (3, 3)
    assert jnp.issubdtype(result.dtype, jnp.complex64)

    # Test case 2: Grazing incidence
    incident_ray = jnp.array([0.0, 1.0, 0.0])
    diffracted_ray = jnp.array([1.0, 0.0, 0.0])
    result = diffraction_coefficients(
        incident_ray, diffracted_ray, edge_vector, k, n, r_prime, r, r0
    )
    assert jnp.allclose(
        result, jnp.zeros((3, 3)), atol=1e-6
    )  # Should be zero for grazing incidence

    # Test case 3: Random inputs
    for _ in range(10):
        incident_ray = random.normal(key, (3,))
        diffracted_ray = random.normal(key, (3,))
        edge_vector = random.normal(key, (3,))
        k = random.uniform(key, (), minval=1, maxval=10)
        n = random.uniform(key, (), minval=1, maxval=2)
        r_prime = random.uniform(key, (), minval=1, maxval=20)
        r = random.uniform(key, (), minval=1, maxval=20)
        r0 = random.uniform(key, (), minval=1, maxval=10)
        result = compute_utd_diffraction_coefficient(
            incident_ray, diffracted_ray, edge_vector, k, n, r_prime, r, r0
        )
        assert result.shape == (3, 3)
        assert jnp.issubdtype(result.dtype, jnp.complex64)

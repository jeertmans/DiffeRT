import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from differt.em.fresnel import fresnel_coefficients

def test_fresnel_coefficients(key: PRNGKeyArray) -> None:
    # Test case 1: Normal incidence
    incident_ray = jnp.array([0.0, 0.0, 1.0])
    reflected_ray = jnp.array([0.0, 0.0, -1.0])
    result = fresnel_coefficients(incident_ray, reflected_ray)
    expected = jnp.array([[-0.2, 0, 0], [0, -0.2, 0], [0, 0, 0]], dtype=complex)
    assert jnp.allclose(result, expected, atol=1e-6)

    # Test case 2: 45-degree incidence
    incident_ray = jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2)
    reflected_ray = jnp.array([1.0, 0.0, -1.0]) / jnp.sqrt(2)
    result = fresnel_coefficients(incident_ray, reflected_ray)
    expected = jnp.array([[-0.1716, 0, 0], [0, -0.2284, 0], [0, 0, 0]], dtype=complex)
    assert jnp.allclose(result, expected, atol=1e-4)

    # Test case 3: Random incidence
    for key_incident in jax.random.split(key, 10):
        incident_ray = jax.random.normal(key_incident, (3,))
        reflected_ray = incident_ray * jnp.array([1, 1, -1])  # Reflect about xy-plane
        result = fresnel_coefficients(incident_ray, reflected_ray)
        assert result.shape == (3, 3)
        assert jnp.issubdtype(result.dtype, complex)
        assert jnp.all(jnp.abs(result) <= 1)  # Coefficients should be <= 1 in magnitude

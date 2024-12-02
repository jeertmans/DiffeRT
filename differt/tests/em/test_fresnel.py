import chex
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from differt.em._fresnel import (
    fresnel_coefficients,
    reflection_coefficients,
    refraction_coefficients,
)


def test_fresnel_coefficients(key: PRNGKeyArray) -> None:
    key_n_1, key_n_2 = jax.random.split(key, 2)

    n_1 = jax.random.uniform(key_n_1, (100,), minval=0.01, maxval=2.0)
    n_2 = jax.random.uniform(key_n_2, (100,), minval=0.01, maxval=2.0)

    n_r = n_2 / n_1
    theta_i = jnp.linspace(0, jnp.pi / 2)
    cos_theta_i = jnp.cos(theta_i)
    n_r = n_r[..., None]
    cos_theta_i = cos_theta_i[None, ...]

    (r_s, r_p), (t_s, t_p) = fresnel_coefficients(n_r, cos_theta_i)

    theta_c = jnp.arcsin(jnp.minimum(n_r, 1.0))

    for array in (r_s, r_p, t_s, t_p):
        chex.assert_tree_all_finite(jnp.where(theta_i <= theta_c, array, 0.0))

    chex.assert_trees_all_equal((r_s, r_p), reflection_coefficients(n_r, cos_theta_i))

    chex.assert_trees_all_equal((t_s, t_p), refraction_coefficients(n_r, cos_theta_i))

    chex.assert_trees_all_close(t_s, r_s + 1, atol=1e-6)
    chex.assert_trees_all_close(n_r * t_p, r_p + 1, atol=1e-6)


def test_reflection_coefficients() -> None:
    n_r = jnp.array(1.5)  # Glass

    # 1. Normal incidence
    cos_theta_i = jnp.array(1.0)

    got_r_s, got_r_p = reflection_coefficients(n_r, cos_theta_i)

    chex.assert_trees_all_equal(got_r_s, -got_r_p)

    # 2. 45-degree incidence
    cos_theta_i = jnp.cos(jnp.pi / 2)

    got_r_s, got_r_p = reflection_coefficients(n_r, cos_theta_i)
    chex.assert_trees_all_close(got_r_s**2, -got_r_p)

    # 3. Brewster's angle
    theta_b = jnp.arctan(n_r)
    cos_theta_i = jnp.cos(theta_b)

    _, got_r_p = reflection_coefficients(n_r, cos_theta_i)

    chex.assert_trees_all_equal(got_r_p, 0 + 0j)

    # 4. Total reflection
    n_r = 1 / jnp.array(1.5)
    theta_i = jnp.arcsin(n_r / 1.0)

    cos_theta_i = jnp.cos(theta_i)

    got_r_s, got_r_p = reflection_coefficients(n_r, cos_theta_i)

    chex.assert_trees_all_equal(got_r_s, 1 + 0j)
    chex.assert_trees_all_equal(got_r_p, 1 + 0j)

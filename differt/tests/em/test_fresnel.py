import chex
import jax.numpy as jnp

from differt.em.fresnel import reflection_coefficients


def test_reflection_coefficients() -> None:
    n_t = jnp.array(1.5)  # Glass
    epsilon_r = (n_t**2).astype(complex)

    # 1. Normal incidence
    cos_theta_i = jnp.array(1.0)

    got_r_s, got_r_p = reflection_coefficients(epsilon_r, cos_theta_i)

    chex.assert_trees_all_equal(got_r_s, -got_r_p)

    # 2. 45-degree incidence
    cos_theta_i = jnp.cos(jnp.pi / 2)

    got_r_s, got_r_p = reflection_coefficients(epsilon_r, cos_theta_i)
    chex.assert_trees_all_close(got_r_s**2, -got_r_p)

    # 3. Brewster's angle
    theta_b = jnp.arctan(n_t)
    cos_theta_i = jnp.cos(theta_b)

    _, got_r_p = reflection_coefficients(epsilon_r, cos_theta_i)

    chex.assert_trees_all_equal(got_r_p, 0 + 0j)

    # 4. Total reflection
    n_t = 1 / jnp.array(1.5)
    epsilon_r = (n_t**2).astype(complex)
    theta_i = jnp.arcsin(n_t / 1.0)

    cos_theta_i = jnp.cos(theta_i)

    got_r_s, got_r_p = reflection_coefficients(epsilon_r, cos_theta_i)

    chex.assert_trees_all_equal(got_r_s, 1 + 0j)
    chex.assert_trees_all_equal(got_r_p, 1 + 0j)

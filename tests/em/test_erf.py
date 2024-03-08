import chex
import jax.numpy as jnp
import numpy as np
import pytest
from chex import Array
from scipy.special import erf as erf_scipy

from differt.em.erf import erf


@pytest.mark.parametrize(
    "z",
    (
        # jnp.linspace(-10.0, +10.0),
        1j * jnp.linspace(+5.0, +10.0),
        # jnp.linspace(-10.0, +10.0) + 1j * jnp.linspace(-10.0, +10.0),
    ),
)
def test_erf(z: Array) -> None:
    got = erf(z)
    expected = jnp.asarray(erf_scipy(np.asarray(z)))
    chex.assert_trees_all_close(got, expected)

import chex
import jax.numpy as jnp
import numpy as np
import scipy.special as sp

from differt.em._utd import F


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

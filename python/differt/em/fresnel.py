"""
Fresnel integrals.

JAX implementation of the SciPy function :py:func:`scipy.special.fresnel`.
"""

import jax
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Num, jaxtyped

from .erf import erf


@jax.jit
@jaxtyped(typechecker=typechecker)
def fresnel(
    z: Num[Array, " *batch"],
) -> tuple[Complex[Array, " *batch"], Complex[Array, " *batch"]]:
    """
    Evaluate the two Fresnel integrals at the given points.

    This current implementation is written using
    the error function :py:func:`erf<differt.em.erf>`
    see :cite:`fresnel-integrals`.

    Args:
        z: The array of real or complex points to evaluate.

    Return:
        A tuple of two arrays, one for each of the Fresnel integrals.

    Examples:
        The following plots the Fresnel for real-valued inputs.

        .. plotly::

            >>> import jax.numpy as jnp
            >>> from differt.em.fresnel import fresnel
            >>>
            >>> t = jnp.linspace(0.0, 5.0)
            >>> s, c = fresnel(t)
            >>> s
    """
    # Constant factors
    sqrtpi_2_4 = 0.31332853432887503  # 0.25 * jnp.sqrt(0.5 * jnp.pi)
    _sqrt2 = 0.7071067811865476  # jnp.sqrt(0.5)

    # Erf function evaluations
    ep = erf((1 + 1j) * _sqrt2 * z)
    em = erf((1 - 1j) * _sqrt2 * z)

    s = sqrtpi_2_4 * (1 + 1j) * (ep - 1j * em)
    c = sqrtpi_2_4 * (1 - 1j) * (ep + 1j * em)

    return s, c

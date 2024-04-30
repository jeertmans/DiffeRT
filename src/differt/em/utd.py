"""
Uniform Theory of Diffraction (UTD) utilities.

The foundamentals of UTD are described in :cite:`utd-mcnamara`.
"""

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Inexact, jaxtyped

from .special import erfc


@jax.jit
@jaxtyped(typechecker=typechecker)
def F(z: Inexact[Array, " *batch"]) -> Complex[Array, " *batch"]:  # noqa: N802
    r"""
    Evaluate the transition function :cite:`utd-mcnamara{p. 184}` at the given points.

    The transition function is defined as follows:

    .. math::
        F(z) = 2j \sqrt{z} e^{j z} \int\limits_\sqrt{z}^\infty e^{-j u^2} \text{d}u,

    where :math:`j^2 = -1`.

    As detailed in :cite:`utd-mcnamara{p. 164}`, the integral can be expressed in
    terms of Fresnel integrals (:math:`C(z)` and :math:`S(z)`), so that:

    .. math::
        C(z) - j S(z) = \int\limits_\sqrt{z}^\infty e^{-j u^2} \text{d}u.

    Because JAX does not provide a XLA implementation of
    :py:data:`scipy.special.fresnel`, this implementation relies on a
    custom complex-valued implementation of the error function and
    the fact that:

    .. math::
        C(z) - j S(z) = \sqrt{\frac{\pi}{2}}\frac{1-j}{2}\text{erf}\left(\frac{1+j}{\sqrt{2}}z\right).

    As a result, we can further simplify :math:`F(z)` to:

    .. math::
        F(z) = \sqrt{\frac{\pi}{2}} \sqrt{z} e^{j z} (1 - j) \text{erfc}\left(\frac{1+j}{\sqrt{2}}z\right),

    where :math:`\text{erfc}` is the complementary error function.

    Args:
        z: The array of real or complex points to evaluate.

    Return:
        The values of the transition function at the given point.

    Examples:
        .. plot::

            The following example reproduces the same plot as in
            :cite:`utd-mcnamara{fig. 4.16}`.

            >>> from differt.em.utd import F
            >>>
            >>> x = jnp.logspace(-3, 1, 100)
            >>> y = F(x)
            >>>
            >>> A = jnp.abs(y)  # Amplitude of F(x)
            >>> P = jnp.angle(y, deg=True)  # Phase (in deg.) of F(x)
            >>>
            >>> fig, ax1 = plt.subplots()
            >>>
            >>> ax1.semilogx(x, A)  # doctest: +SKIP
            >>> ax1.set_xlabel("$x$")  # doctest: +SKIP
            >>> ax1.set_ylabel("Magnitude - solid line")  # doctest: +SKIP
            >>> ax2 = plt.twinx()
            >>> ax2.semilogx(x, P, "--")  # doctest: +SKIP
            >>> ax2.set_ylabel("Phase (°) - dashed line")  # doctest: +SKIP
            >>> plt.tight_layout()  # doctest: +SKIP
    """
    factor = jnp.sqrt(jnp.pi / 2)
    sqrt_z = jnp.sqrt(z)

    return (
        (1 + 1j)
        * factor
        * sqrt_z
        * jnp.exp(1j * z)
        * erfc((1 + 1j) * sqrt_z / jnp.sqrt(2))
    )

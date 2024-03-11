"""
Special functions.

This module extends the :py:mod:`jax.scipy.special` module
by adding missing function from :py:mod:`scipy.special`,
or by extending already implemented function to the
complex domain.

Those new implementation are needed to keep the ability
of differentating code, otherwise we could just
call the SciPy function and wrap their output
with :py:func:`jnp.asarray<jax.numpy.asarray>`.
"""

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax.scipy.special import erf as erfx
from jaxtyping import Array, Inexact, jaxtyped


# @jax.jit
@jaxtyped(typechecker=typechecker)
def erf(z: Inexact[Array, " *batch"], steps: int = 20) -> Inexact[Array, " *batch"]:
    """
    Evaluate the error function at the given points.

    This current implementation is written using
    the error function :py:func:`erf<jax.scipy.special.erf>`
    and the approximation as detailed in <TODO>.

    The output type (real or complex) is determined by the
    input type.

    Args:
        z: The array of real or complex points to evaluate.
        steps: The number of steps used to approximate the
            infinite sum. Not used if the input array
            is not complex.

    Return:
        The values of the error function at the given point.

    Examples:
        The following plots the error function for real-valued inputs.

        .. plot::

            >>> from differt.em.special import erf
            >>>
            >>> x = jnp.linspace(-3.0, +3.0)
            >>> y = erf(x)
            >>> plt.plot(x, y.real)

        The following plots the error function for complex-valued inputs.

        .. plotly::

            >>> from differt.em.special import erf
            >>>
            >>> x = y = jnp.linspace(-2.0, +2.0, 200)
            >>> a, b = jnp.meshgrid(x, y)
            >>> z = erf(a + 1j * b)
            >>> go.Figure(data=[go.Surface(x=x, y=y, z=jnp.abs(z), surfacecolor=jnp.angle(z))])
    """
    if jnp.issubdtype(z.dtype, jnp.floating):
        return erfx(z)
    # https://math.stackexchange.com/questions/712434/erfaib-error-function-separate-into-real-and-imaginary-part#comment1491304_712568
    # https://granite.phys.s.u-tokyo.ac.jp/svn/LCGT/trunk/sensitivity/Matlab/bKAGRA/@double/erfz.pdf
    # TODO: fixme
    x, y = z.real, z.imag

    print(f"{x = }")
    print(f"{y = }")

    two_x = 2 * x
    two_x_y = two_x * y
    x_squared = x * x
    four_x_squared = x_squared
    cos_two_x_y = jnp.cos(two_x_y)
    sin_two_x_y = jnp.sin(two_x_y)

    exp = jnp.exp(-x_squared)

    def scan_fun(carry_sum, k):
        k_squared = k * k
        k_y = k * y
        print(f"{k = }")
        print(f"{k_y = }")
        cosh_k_y = jnp.cosh(k_y)
        sinh_k_y = jnp.sinh(k_y)
        cosh_k_y = 1.0
        #sinh_k_y = 1.0


        factor = jnp.exp(-k_squared * 0.25) / (k_squared + four_x_squared)
        factor = 1.0

        f = two_x * (1 - cos_two_x_y * cosh_k_y) + k * cos_two_x_y * sinh_k_y
        g = two_x * sin_two_x_y * cosh_k_y + k * cos_two_x_y * sinh_k_y

        return carry_sum + factor * (f + 1j * g), None

    print(z.dtype)

    carry_sum = jnp.zeros_like(z)

    for k in jnp.arange(1.0, steps + 1.0, dtype=z.dtype):
        carry_sum, _ = scan_fun(carry_sum, k)

    """
    carry_sum = jax.lax.scan(
        scan_fun,
        init=jnp.zeros_like(z),
        xs=jnp.arange(1.0, steps + 1.0, dtype=z.dtype),
    )[0]
    """

    print("after carry sum")

    # e = (exp_one * (1.0 - exp_two)) / (jnp.pi * two_x)

    # e = jnp.where(x == 0.0, 1j * x / jnp.pi, e)

    x_non_zero = jnp.where(x == 0.0, 1.0, x)
    a = jnp.where(
        x == 0.0,
        1j * y / jnp.pi,
        ((1 - cos_two_x_y) + 1j * sin_two_x_y) * exp / (2 * jnp.pi * x_non_zero),
    )  # Fix limit x -> 0
    print(a)
    b = carry_sum * exp * 2 / jnp.pi

    return erfx(x) + a + b


@jax.jit
@jaxtyped(typechecker=typechecker)
def fresnel(
    z: Inexact[Array, " *batch"],
) -> tuple[Inexact[Array, " *batch"], Inexact[Array, " *batch"]]:
    """
    Evaluate the two Fresnel integrals at the given points.

    This current implementation is written using
    the error function :py:func:`erf<differt.em.special.erf>`
    see :cite:`fresnel-integrals`.

    The output type (real or complex) is determined by the
    input type.

    Args:
        z: The array of real or complex points to evaluate.

    Return:
        A tuple of two arrays, one for each of the Fresnel integrals.

    Examples:
        The following plots the Fresnel for real-valued inputs.

        .. plot::

            >>> from differt.em.special import fresnel
            >>>
            >>> t = jnp.linspace(0.0, 5.0, 200)
            >>> s, c = fresnel(t)
            >>> plt.plot(t, s.real, t, c.real)
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

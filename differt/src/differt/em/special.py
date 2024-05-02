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


@jax.jit
@jaxtyped(typechecker=typechecker)
def erf(z: Inexact[Array, " *batch"]) -> Inexact[Array, " *batch"]:
    r"""
    Evaluate the error function at the given points.

    The current implementation is written using
    the real-valued error function :py:func:`jax.scipy.special.erf`
    and the approximation as detailed in :cite:`erf-complex`.

    The output type (real or complex) is determined by the
    input type.

    Warning:
        Currently, we observe that
        this function and :py:data:`scipy.special.erf`
        starts to diverge for :math:`|z| > 6`. If you know
        how to avoid this problem, please contact us!

    Args:
        z: The array of real or complex points to evaluate.

    Return:
        The values of the error function at the given point.

    Notes:
        Regarding performances, there are two possible outputs:

        1. If ``z`` is real, then this function compiles to
           :py:func:`jax.scipy.special.erf`, and will therefore
           have the same performances (when JIT compilation
           is done). Compared to the SciPy equivalent, we measured
           that our implementation is **~ 10 times faster**.
        2. If ``z`` is complex, then our implementation is
           **~ 3 times faster** than
           :py:data:`scipy.special.erf`.

        Those results were measured on centered random uniform arrays
        with :math:`10^5` elements.

    Examples:
        The following plots the error function for real-valued inputs.

        .. plot::

            >>> from differt.em.special import erf
            >>>
            >>> x = jnp.linspace(-3.0, +3.0)
            >>> y = erf(x)
            >>> plt.plot(x, y.real)  # doctest: +SKIP
            >>> plt.xlabel("$x$")  # doctest: +SKIP
            >>> plt.ylabel(r"$\text{erf}(x)$")  # doctest: +SKIP

        The following plots the error function for complex-valued inputs.

        .. plotly::

            >>> from differt.em.special import erf
            >>> from scipy.special import erf
            >>>
            >>> x = y = jnp.linspace(-2.0, +2.0, 200)
            >>> a, b = jnp.meshgrid(x, y)
            >>> z = erf(a + 1j * b)
            >>> fig = go.Figure(
            ...     data=[
            ...         go.Surface(
            ...             x=x,
            ...             y=y,
            ...             z=jnp.abs(z),
            ...             colorscale="phase",
            ...             surfacecolor=jnp.angle(z),
            ...             colorbar=dict(title="Arg(erf(z))"),
            ...         )
            ...     ]
            ... )
            >>> fig.update_layout(
            ...     scene=dict(
            ...         xaxis=dict(title="Re(z)"),
            ...         yaxis=dict(title="Im(z)"),
            ...         zaxis=dict(title="Abs(erf(z))"),
            ...     )
            ... )  # doctest: +SKIP
            >>> fig  # doctest: +SKIP
    """
    if jnp.issubdtype(z.dtype, jnp.floating):
        return erfx(z)
    # https://granite.phys.s.u-tokyo.ac.jp/svn/LCGT/trunk/sensitivity/Matlab/bKAGRA/@double/erfz.pdf

    if jnp.issubdtype(z.dtype, jnp.complex128):  # double precision
        N = 13  # noqa: N806
        M = 14  # noqa: N806
    else:  # single precision
        N = 9  # noqa: N806
        M = 10  # noqa: N806

    r = z.real
    i = jnp.abs(z.imag)
    r_squared = r * r

    exp_r_squared = jnp.exp(-r_squared)
    exp_2j_r_i = jnp.exp(-2j * r * i)

    f_sum = jnp.zeros_like(z)
    g_sum = jnp.zeros_like(z)
    h_sum = jnp.zeros_like(z)

    for n in range(1, N + 1):
        n_squared = n * n
        n_squared_over_four = n_squared / 4
        den = 1 / (n_squared_over_four + r_squared)
        exp_f = jnp.exp(-n_squared_over_four)
        exp_g = jnp.exp(+n * i - n_squared_over_four)
        exp_h = jnp.exp(-n * i - n_squared_over_four)

        f_sum += exp_f * den
        g_sum += exp_g * (r - 1j * n / 2) * den
        h_sum += exp_h * (r + 1j * n / 2) * den

    for n in range(N + 1, N + M + 1):
        n_squared = n * n
        n_squared_over_four = n_squared / 4
        exp_g = jnp.exp(+n * i - n_squared_over_four)

        g_sum += exp_g * (r - 1j * n / 2) / (n_squared_over_four + r_squared)

    r_non_zero = jnp.where(r == 0.0, 1.0, r)
    e = jnp.where(
        r == 0.0,
        1j * i / jnp.pi,
        (exp_r_squared * (1 - exp_2j_r_i)) / (2 * jnp.pi * r_non_zero),
    )  # Fixes limit r -> 0
    f = r * exp_r_squared * f_sum / jnp.pi
    g = exp_r_squared * g_sum / (2 * jnp.pi)
    h = exp_r_squared * h_sum / (2 * jnp.pi)

    res = erfx(r) + e + f - exp_2j_r_i * (g + h)
    res = jnp.where(z.imag < 0, jnp.conj(res), res)
    return res


@jax.jit
@jaxtyped(typechecker=typechecker)
def erfc(z: Inexact[Array, " *batch"]) -> Inexact[Array, " *batch"]:
    r"""
    Evaluate the complementary error function at the given points.

    The output type (real or complex) is determined by the
    input type.

    See :py:func:`erf` for more details.

    Args:
        z: The array of real or complex points to evaluate.

    Return:
        The values of the complementary error function at the given point.

    Examples:
        The following plots the complementary error function for real-valued inputs.

        .. plot::

            >>> from differt.em.special import erfc
            >>>
            >>> x = jnp.linspace(-3.0, +3.0)
            >>> y = erfc(x)
            >>> plt.plot(x, y.real)  # doctest: +SKIP
            >>> plt.xlabel("$x$")  # doctest: +SKIP
            >>> plt.ylabel(r"$\text{erfc}(x)$")  # doctest: +SKIP
    """
    return 1.0 - erf(z)


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
            >>> plt.plot(t, s.real, label=r"$y=S(x)$")  # doctest: +SKIP
            >>> plt.plot(t, c.real, "--", label=r"$y=C(x)$")  # doctest: +SKIP
            >>> plt.xlabel("$x$")  # doctest: +SKIP
            >>> plt.ylabel("$y$")  # doctest: +SKIP
            >>> plt.legend()  # doctest: +SKIP
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

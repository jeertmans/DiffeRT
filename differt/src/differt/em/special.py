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
from jaxtyping import Array, Float, Inexact, jaxtyped


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

    Returns:
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
    if jnp.isrealobj(z):
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
    return jnp.where(z.imag < 0, jnp.conj(res), res)


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

    Returns:
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

    This real-valued implementation is based on the SciPy C++
    implementation, see
    `here <https://github.com/scipy/scipy/blob/1abc9faf43d7b82e0b26589bb705d8d009b18713/scipy/special/special/cephes/fresnl.h>`_, and should be very accurate.

    For complex-valued arguments, it uses
    :func:`erfc`, see :cite:`fresnel-integrals`, and
    the accuracy is observed to be much lower than that
    of real-valued arguments.

    The output type (real or complex) is determined by the
    input type.

    Args:
        z: The array of real or complex points to evaluate.

    Returns:
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
    if jnp.iscomplexobj(z):
        # Constant factors
        sqrtpi_2_4 = 0.31332853432887503  # 0.25 * jnp.sqrt(0.5 * jnp.pi)
        _sqrt2 = 0.7071067811865476  # jnp.sqrt(0.5)

        # Erf function evaluations
        ep = erf((1 + 1j) * _sqrt2 * z)
        em = erf((1 - 1j) * _sqrt2 * z)

        s = sqrtpi_2_4 * (1 + 1j) * (ep - 1j * em)
        c = sqrtpi_2_4 * (1 - 1j) * (ep + 1j * em)

        return s, c

    # This part is mostly a direct translation of SciPy's C++ code.

    xxa = z

    fresnl_sn = jnp.array(
        [
            -2.99181919401019853726e3,
            +7.08840045257738576863e5,
            -6.29741486205862506537e7,
            +2.54890880573376359104e9,
            -4.42979518059697779103e10,
            +3.18016297876567817986e11,
        ]
    )

    fresnl_sd = jnp.array(
        [
            +1.00000000000000000000e0,
            +2.81376268889994315696e2,
            +4.55847810806532581675e4,
            +5.17343888770096400730e6,
            +4.19320245898111231129e8,
            +2.24411795645340920940e10,
            +6.07366389490084639049e11,
        ]
    )

    fresnl_cn = jnp.array(
        [
            -4.98843114573573548651e-8,
            +9.50428062829859605134e-6,
            -6.45191435683965050962e-4,
            +1.88843319396703850064e-2,
            -2.05525900955013891793e-1,
            +9.99999999999999998822e-1,
        ]
    )

    fresnl_cd = jnp.array(
        [
            +3.99982968972495980367e-12,
            +9.15439215774657478799e-10,
            +1.25001862479598821474e-7,
            +1.22262789024179030997e-5,
            +8.68029542941784300606e-4,
            +4.12142090722199792936e-2,
            +1.00000000000000000118e0,
        ]
    )

    fresnl_fn = jnp.array(
        [
            +4.21543555043677546506e-1,
            +1.43407919780758885261e-1,
            +1.15220955073585758835e-2,
            +3.45017939782574027900e-4,
            +4.63613749287867322088e-6,
            +3.05568983790257605827e-8,
            +1.02304514164907233465e-10,
            +1.72010743268161828879e-13,
            +1.34283276233062758925e-16,
            +3.76329711269987889006e-20,
        ]
    )

    fresnl_fd = jnp.array(
        [
            +1.00000000000000000000e0,
            +7.51586398353378947175e-1,
            +1.16888925859191382142e-1,
            +6.44051526508858611005e-3,
            +1.55934409164153020873e-4,
            +1.84627567348930545870e-6,
            +1.12699224763999035261e-8,
            +3.60140029589371370404e-11,
            +5.88754533621578410010e-14,
            +4.52001434074129701496e-17,
            +1.25443237090011264384e-20,
        ]
    )

    fresnl_gn = jnp.array(
        [
            +5.04442073643383265887e-1,
            +1.97102833525523411709e-1,
            +1.87648584092575249293e-2,
            +6.84079380915393090172e-4,
            +1.15138826111884280931e-5,
            +9.82852443688422223854e-8,
            +4.45344415861750144738e-10,
            +1.08268041139020870318e-12,
            +1.37555460633261799868e-15,
            +8.36354435630677421531e-19,
            +1.86958710162783235106e-22,
        ]
    )

    fresnl_gd = jnp.array(
        [
            +1.00000000000000000000e0,
            +1.47495759925128324529e0,
            +3.37748989120019970451e-1,
            +2.53603741420338795122e-2,
            +8.14679107184306179049e-4,
            +1.27545075667729118702e-5,
            +1.04314589657571990585e-7,
            +4.60680728146520428211e-10,
            +1.10273215066240270757e-12,
            +1.38796531259578871258e-15,
            +8.39158816283118707363e-19,
            +1.86958710162783236342e-22,
        ]
    )

    @jax.jit
    def sincospi(
        x: Float[Array, " *batch"],
    ) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
        """
        Accurate evaluation of sin(pi * x) and cos(pi * x).

        As based on the sinpi and cospi functions from SciPy, see:
        - https://github.com/scipy/scipy/blob/ae7e7c7109d957bb3c9798497d35e2fe65cd78be/scipy/special/special/cephes/trig.h
        """
        s = jnp.sign(x)
        x = jnp.abs(x)
        r = jnp.fmod(x, 2.0)

        sinpi = jnp.where(
            r < 0.5,
            s * jnp.sin(jnp.pi * r),
            jnp.where(
                r > 1.5,
                s * jnp.sin(jnp.pi * (r - 2.0)),
                -s * jnp.sin(jnp.pi * (r - 1.0)),
            ),
        )
        cospi = jnp.where(
            r == 0.5,
            0.0,
            jnp.where(
                r < 1.0, -jnp.sin(jnp.pi * (r - 0.5)), jnp.sin(jnp.pi * (r - 1.5))
            ),
        )

        return sinpi, cospi

    x = jnp.abs(xxa)

    x2 = x * x

    # Infinite x values
    s_inf = c_inf = 0.5

    # Small x values
    t = x2 * x2
    s_small = x * x2 * jnp.polyval(fresnl_sn[:6], t) / jnp.polyval(fresnl_sd[:7], t)
    c_small = x * jnp.polyval(fresnl_cn[:6], t) / jnp.polyval(fresnl_cd[:7], t)

    # Large x values
    sinpi, cospi = sincospi(x2 / 2)
    c_large = 0.5 + 1 / (jnp.pi * x) * sinpi
    s_large = 0.5 - 1 / (jnp.pi * x) * cospi

    # Other x values
    t = jnp.pi * x2
    u = 1.0 / (t * t)
    t = 1.0 / t
    f = 1.0 - u * jnp.polyval(fresnl_fn, u) / jnp.polyval(fresnl_fd, u)
    g = t * jnp.polyval(fresnl_gn, u) / jnp.polyval(fresnl_gd, u)

    t = jnp.pi * x
    c_other = 0.5 + (f * sinpi - g * cospi) / t
    s_other = 0.5 - (f * cospi + g * sinpi) / t

    isinf = jnp.isinf(xxa)
    small = x2 < 2.5625
    large = x > 36974.0
    s = jnp.where(
        isinf, s_inf, jnp.where(small, s_small, jnp.where(large, s_large, s_other))
    )
    c = jnp.where(
        isinf, c_inf, jnp.where(small, c_small, jnp.where(large, c_large, c_other))
    )

    neg = xxa < 0.0
    s = jnp.where(neg, -s, s)
    c = jnp.where(neg, -c, c)

    return s, c

"""
Error function evaluated on complex values.
"""

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax.scipy.special import erf as erfx
from jaxtyping import Array, Complex, Num, jaxtyped


@jax.jit
@jaxtyped(typechecker=typechecker)
def erf(z: Num[Array, " *batch"], steps: int = 100) -> Complex[Array, " *batch"]:
    """
    Evaluate the error function at the given points.

    This current implementation is written using
    the error function :py:func:`erf<jax.scipy.special.erf>`
    and the approximation as detail in <TODO>.

    Args:
        z: The array of real or complex points to evaluate.
        steps: TODO.

    Return:
        The values of the error function at the given point.

    Examples:
        The following plots the error function for real-valued inputs.

        .. plot::

            >>> from differt.em.erf import erf
            >>>
            >>> x = jnp.linspace(-3.0, +3.0)
            >>> y = erf(x)
            >>> plt.plot(x, y.real)

        .. plotly::

            >>> from differt.em.erf import erf
            >>>
            >>> x = y = jnp.linspace(-2.0, +2.0)
            >>> a, b = jnp.meshgrid(x, y)
            >>> z = erf(a + 1j * b)
            >>> go.Figure(data=[go.Surface(x=x, y=y, z=jnp.abs(x), surfacecolor=jnp.angle(z))])
    """
    # https://math.stackexchange.com/questions/712434/erfaib-error-function-separate-into-real-and-imaginary-part#comment1491304_712568
    # https://granite.phys.s.u-tokyo.ac.jp/svn/LCGT/trunk/sensitivity/Matlab/bKAGRA/@double/erfz.pdf
    x, y = z.real, z.imag

    two_x = 2 * x
    two_x_y = two_x * y
    x_squared = x * x
    four_x_squared = x_squared

    exp_one = jnp.exp(-x_squared)
    exp_two = jnp.exp(-2j * two_x_y)

    def scan_fun(carry_sum, k):
        k_squared = k * k
        k_y = k * y
        cosh_k_y = jnp.cosh(k_y)
        sinh_k_y = jnp.sinh(k_y)

        print(f"{k = }")

        f1 = jnp.exp(-k_squared * 0.25) / (k_squared + four_x_squared)
        f2 = two_x - exp_two * (two_x * cosh_k_y - 1j * k * sinh_k_y)

        return carry_sum + f1 * f2, None

    carry_sum = jax.lax.scan(
        scan_fun,
        init=jnp.zeros_like(z, dtype=jnp.complex64),
        xs=jnp.arange(1.0, steps + 1.0),
    )[0]

    e = (exp_one * (1.0 - exp_two)) / (jnp.pi * two_x)

    e = jnp.where(x == 0.0, 1j * x / jnp.pi, e)

    s = 2 * exp_one * carry_sum / jnp.pi

    return erfx(x) + e + s

"""Fresnel coefficients utilities."""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, Inexact, jaxtyped

from differt.utils import safe_divide


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def refractive_indices(
    epsilon_r: Inexact[ArrayLike, " *#batch"],
    mu_r: Inexact[ArrayLike, " *#batch"] | None = None,
) -> Inexact[Array, " *batch"]:
    r"""
    Compute the refractive indices corresponding to relative permittivities and relative permeabilities.

    The refractive index :math:`n` is simply defined as

    .. math::
        n = \sqrt{\epsilon_r\mu_r},

    where :math:`\epsilon_r` is the relative permittivity, and :math:`\mu_r` is the relative permeability.

    Args:
        epsilon_r: The relative refractive indices.

            This is the ratios of the refractive indices of the second
            media over the refractive indices of the first media.
        mu_r: The relative permeabilities. If not provided,
            a value of 1 is used.

    Returns:
        The array of refractive indices.

        The output dtype will only be complex if any of the provided arguments
        has a complex dtype.

    .. seealso::

        :func:`fresnel_coefficients`

        :func:`reflection_coefficients`

        :func:`refraction_coefficients`
    """
    sqrt_n = epsilon_r if mu_r is None else epsilon_r * mu_r
    return jax.lax.integer_pow(sqrt_n, 2)


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def fresnel_coefficients(
    n_r: Inexact[ArrayLike, " *#batch"],
    cos_theta_i: Float[Array, " *#batch"],
) -> tuple[
    tuple[Inexact[Array, " *batch"], Inexact[Array, " *batch"]],
    tuple[Inexact[Array, " *batch"], Inexact[Array, " *batch"]],
]:
    r"""
    Compute the Fresnel reflection and refraction coefficients at an interface.

    The Snell's law describes the relationship between the angles of incidence
    and refraction:

    .. math::
        n_i\sin\theta_i = n_t\sin\theta_t,

    where :math:`n` is the refraction index, :math:`\theta` is the angle of between the ray path
    and the normal to the interface, and :math:`i` and :math:`t` indicate,
    respectively, the first (i.e., incidence) and the second (i.e., transmission)
    media.

    The s and p reflection coefficients are:

    .. math::
        r_s = \frac{n_i\cos\theta_i - n_t\cos\theta_t}{n_i\cos\theta_i + n_t\cos\theta_t},

    and

    .. math::
        r_p = \frac{n_t\cos\theta_i - n_i\cos\theta_t}{n_t\cos\theta_i + n_i\cos\theta_t}.

    The s and p refraction coefficients are:

    .. math::
        t_s = \frac{2n_i\cos\theta_i}{n_i\cos\theta_i + n_t\cos\theta_t},

    and

    .. math::
        t_p = \frac{2n_i\cos\theta_i}{n_t\cos\theta_i + n_i\cos\theta_t}.

    Then, we define :math:`n_r \triangleq \frac{n_t}{n_i}` and rewrite the four coefficients as:

    .. math::
        r_s &= \frac{\cos\theta_i - n_r\cos\theta_t}{\cos\theta_i + n_r\cos\theta_t},\\
        r_p &= \frac{n_r^2\cos\theta_i - n_r\cos\theta_t}{n_r^2\cos\theta_i + n_r\cos\theta_t},\\
        t_s &= \frac{2\cos\theta_i}{\cos\theta_i + n_r\cos\theta_t},\\
        t_p &= \frac{2n_r\cos\theta_i}{n_r^2\cos\theta_i + n_r\cos\theta_t},

    where :math:`n_t\cos\theta_t` is obtained from:

    .. math::
        n_r\cos\theta_t = \sqrt{n_r^2 + \cos^2\theta_i - 1}.

    Args:
        n_r: The relative refractive indices.

            This is the ratios of the refractive indices of the second
            media over the refractive indices of the first media.
        cos_theta_i: The (cosine of the) angles of incidence (or reflection).

    Returns:
        The reflection and refraction coefficients for s and p polarizations.

        The output dtype will only be complex if any of the provided arguments
        has a complex dtype.

    .. seealso::

        :func:`reflection_coefficients`

        :func:`refraction_coefficients`

        :func:`refractive_indices`

    Examples:
        .. plot::

            The following example reproduces the air-to-glass Fresnel coefficient.
            The Brewster angle (defined by :math:`r_p=0`) is indicated by the vertical
            red line.

            >>> from differt.em import fresnel_coefficients
            >>>
            >>> n = 1.5  # Air to glass
            >>> theta = jnp.linspace(0, jnp.pi / 2)
            >>> cos_theta = jnp.cos(theta)
            >>> (r_s, r_p), (t_s, t_p) = fresnel_coefficients(n, cos_theta)
            >>> theta_d = jnp.rad2deg(theta)
            >>> theta_b = jnp.rad2deg(jnp.arctan(n))
            >>> plt.plot(theta_d, r_s, "b:", label=r"$r_s$")  # doctest: +SKIP
            >>> plt.plot(theta_d, r_p, "r:", label=r"$r_p$")  # doctest: +SKIP
            >>> plt.plot(theta_d, t_s, "b-", label=r"$t_s$")  # doctest: +SKIP
            >>> plt.plot(theta_d, t_p, "r-", label=r"$t_p$")  # doctest: +SKIP
            >>> plt.axvline(theta_b, color="r", linestyle="--")  # doctest: +SKIP
            >>> plt.xlabel("Angle of incidence (°)")  # doctest: +SKIP
            >>> plt.ylabel("Amplitude")  # doctest: +SKIP
            >>> plt.xlim(0, 90)  # doctest: +SKIP
            >>> plt.ylim(-1.0, 1.0)  # doctest: +SKIP
            >>> plt.title("Fresnel coefficients")  # doctest: +SKIP
            >>> plt.legend()  # doctest: +SKIP
            >>> plt.tight_layout()  # doctest: +SKIP

        .. plot::

            The following example produces the same but glass-to-air interface.
            The critical angle (total internal reflection) is indicated by the vertical
            black line.

            >>> from differt.em import fresnel_coefficients
            >>>
            >>> n = 1/ 1.5  #  Glass to air
            >>> theta = jnp.linspace(0, jnp.pi / 2, 300)
            >>> cos_theta = jnp.cos(theta)
            >>> (r_s, r_p), (t_s, t_p) = fresnel_coefficients(n, cos_theta)
            >>> theta_d = jnp.rad2deg(theta)
            >>> theta_b = jnp.rad2deg(jnp.arctan(n))
            >>> theta_c = jnp.rad2deg(jnp.arcsin(n))
            >>> plt.plot(theta_d, r_s, "b:", label=r"$r_s$")  # doctest: +SKIP
            >>> plt.plot(theta_d, r_p, "r:", label=r"$r_p$")  # doctest: +SKIP
            >>> plt.plot(theta_d, t_s, "b-", label=r"$t_s$")  # doctest: +SKIP
            >>> plt.plot(theta_d, t_p, "r-", label=r"$t_p$")  # doctest: +SKIP
            >>> plt.axvline(theta_b, color="r", linestyle="--")  # doctest: +SKIP
            >>> plt.axvline(theta_c, color="k", linestyle="--")  # doctest: +SKIP
            >>> plt.xlabel("Angle of incidence (°)")  # doctest: +SKIP
            >>> plt.ylabel("Amplitude")  # doctest: +SKIP
            >>> plt.xlim(0, 90)  # doctest: +SKIP
            >>> plt.ylim(-0.5, 3.0)  # doctest: +SKIP
            >>> plt.title("Fresnel coefficients")  # doctest: +SKIP
            >>> plt.legend()  # doctest: +SKIP
            >>> plt.tight_layout()  # doctest: +SKIP
    """
    n_r_squared = jax.lax.integer_pow(n_r, 2)
    cos_theta_i_squared = jax.lax.integer_pow(cos_theta_i, 2)
    n_r_squared_cos_theta_i = n_r_squared * cos_theta_i
    n_r_cos_theta_t = jnp.sqrt(n_r_squared + cos_theta_i_squared - 1)
    two_cos_theta_i = 2 * cos_theta_i

    r_s = safe_divide(
        cos_theta_i - n_r_cos_theta_t,
        cos_theta_i + n_r_cos_theta_t,
    )
    t_s = safe_divide(
        two_cos_theta_i,
        cos_theta_i + n_r_cos_theta_t,
    )
    r_p = safe_divide(
        n_r_squared_cos_theta_i - n_r_cos_theta_t,
        n_r_squared_cos_theta_i + n_r_cos_theta_t,
    )
    t_p = safe_divide(
        n_r * two_cos_theta_i,
        n_r_squared_cos_theta_i + n_r_cos_theta_t,
    )

    return (r_s, r_p), (t_s, t_p)


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def reflection_coefficients(
    n_r: Inexact[ArrayLike, " *#batch"],
    cos_theta_i: Float[Array, " *#batch"],
) -> tuple[Inexact[Array, " *batch"], Inexact[Array, " *batch"]]:
    """
    Compute the Fresnel reflection coefficients at an interface.

    Args:
        n_r: The relative refractive indices.

            This is the ratios of the refractive indices of the second
            media over the refractive indices of the first media.
        cos_theta_i: The (cosine of the) angles of incidence (or reflection).

    Returns:
        The reflection coefficients for s and p polarizations.

        The output dtype will only be complex if any of the provided arguments
        has a complex dtype.

    .. seealso::

        :func:`fresnel_coefficients`

        :func:`refraction_coefficients`

        :func:`refractive_indices`
    """
    return fresnel_coefficients(n_r, cos_theta_i)[0]


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def refraction_coefficients(
    n_r: Inexact[ArrayLike, " *#batch"],
    cos_theta_i: Float[Array, " *#batch"],
) -> tuple[Inexact[Array, " *batch"], Inexact[Array, " *batch"]]:
    """
    Compute the Fresnel refraction coefficients at an interface.

    Args:
        n_r: The relative refractive indices.

            This is the ratios of the refractive indices of the second
            media over the refractive indices of the first media.
        cos_theta_i: The (cosine of the) angles of incidence (or reflection).

    Returns:
        The refraction coefficients for s and p polarizations.

        The output dtype will only be complex if any of the provided arguments
        has a complex dtype.

    .. seealso::

        :func:`fresnel_coefficients`

        :func:`reflection_coefficients`

        :func:`refractive_indices`
    """
    return fresnel_coefficients(n_r, cos_theta_i)[1]

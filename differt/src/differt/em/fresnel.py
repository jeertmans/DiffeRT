r"""
Fresnel coefficients for reflection, as described by the Geometrical Optics (GO).

Note:
    Refraction coefficients are not yet implemented.

As detailed in :cite:`utd-mcnamara{eq. 3.199}`, the GO reflected field
from a smooth conducting surface can be expressed as:

.. math::
    \boldsymbol{E}^r(P) = \boldsymbol{E}^r(Q_r) \sqrt{\frac{\rho_1^r\rho_2^r}{\left(\rho_1^r+s^r\right)\left(\rho_2^r+s^r\right)}} e^{-jks^r},

where :math:`P` is the observation point and :math:`Q_r` is the reflection point on the surface, :math:`\rho_1^r` and :math:`\rho_2^r` are the principal radii of curvature at :math:`Q_r` of the reflected wavefront, :math:`k` is the wavenumber, and :math:`s_r` is the distance between :math:`Q_r` and :math:`P`. Moreover, :math:`\boldsymbol{E}^r(Q_r)` can be expressed in terms of the incident field :math:`\boldsymbol{E}^i`:

.. math::
    \boldsymbol{E}^r(Q_r) = \boldsymbol{E}^r(Q_r) \cdot R

where :math:`\boldsymbol{R}` is the dyadic matrix with the reflection coefficients.


Examples:
    .. plot::

        The following example show how to compute interfence
        patterns from line of sight and reflection on a metal
        ground.

        >>> from differt.em.fresnel import reflection_coefficients
        >>> from differt.em.constants import *
        >>> from differt.geometry.utils import assemble_paths, normalize, path_lengths
        >>> from differt.rt.image_method import image_method
        >>>
        >>> def field(r):
        ...     theta = jnp.arctan2(r[..., 1], r[..., 0])[..., None]
        ...     k_dir, r = normalize(r, keepdims=True)
        ...     e_dir = k_dir.at[..., 0].multiply(-1)[..., [1, 0, 2]]
        ...     return e_dir * jnp.sin(theta) * jnp.sin(r) / r
        >>> tx_position = jnp.array([0.0, 2.0, 0.0])
        >>> rx_position = jnp.array([0.0, 1.0, 0.0])
        >>> n = 1000
        >>> x = jnp.linspace(1, 10, n)
        >>> rx_positions = jnp.tile(rx_position, (n, 1)).at[..., 0].add(x)
        >>> E_los = field(rx_positions - tx_position)
        >>> plt.plot(
        ...     x,
        ...     20 * jnp.log10(jnp.linalg.norm(E_los, axis=-1)),
        ...     label=r"$E_\text{los}$",
        ... )  # doctest: +SKIP
        >>>
        >>> ground_vertex = jnp.array([0.0, 0.0, 0.0])
        >>> ground_normal = jnp.array([0.0, 1.0, 0.0])
        >>> reflection_points = image_method(
        ...     tx_position, rx_positions, ground_vertex, groun_normal
        ... )
        >>> E_at_rp = field(reflection_points - tx_position)
        >>> incident_vectors = normalize(reflection_points - tx_position)[0]
        >>> cos_theta = jnp.dot(ground_normal, -incident_vectors)
        >>> epsilon_r = 1.0
        >>> r_s, r_p = reflection_coefficients(epsilon_r, cos_theta)
        >>> # theta_d = jnp.rad2deg(theta)
        >>> plt.xlabel("Distance to transmitter on x-axis")  # doctest: +SKIP
        >>> plt.ylabel("Received power")  # doctest: +SKIP
        >>> plt.legend()  # doctest: +SKIP
        >>> plt.tight_layout()  # doctest: +SKIP
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Inexact, jaxtyped

from differt.utils import safe_divide


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def reflection_coefficients(
    epsilon_r: Inexact[Array, " *#batch"],
    cos_theta_i: Float[Array, " *#batch"],
    mu_r: Inexact[Array, " *#batch"] | None = None,
) -> tuple[Inexact[Array, " *batch"], Inexact[Array, " *batch"]]:
    r"""
    Compute the Fresnel reflection coefficients for air-to-dielectric interface.

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

    Because we assume the first medium is always air, we have that
    :math:`n_i = 1`, which simplifies the Snell's formula to:

    .. math::
        \sin\theta_i = n_t \sin\theta_t,

    where :math:`n_t = \sqrt{\epsilon_r\mu_r}`, provided as argument of this function.

    Hence, we can rewrite :math:`n_t\cos_t` as:

    .. math::
        n_t\cos\theta_t = \sqrt{\epsilon_r\mu_r + \cos^2\theta_i - 1}.

    Args:
        epsilon_r: The relative permittivities.
        cos_theta_i: The (cosine of the) angles of incidence (or reflection).
        mu_r: The relative permeabilities. If not provided,
            a value of 1 is used.

    Returns:
        The reflection coefficients for s and p polarizations.

        The output dtype will only be complex if any of the provided arguments
        has a complex dtype.

    Examples:
        .. plot::

            The following example reproduces the air-to-glass reflectance
            power coefficient.

            >>> from differt.em.fresnel import reflection_coefficients
            >>>
            >>> n = 1.5  # Air to glass
            >>> epsilon_r = jnp.sqrt(n)
            >>> theta = jnp.linspace(0, jnp.pi / 2)
            >>> cos_theta = jnp.cos(theta)
            >>> r_s, r_p = reflection_coefficients(epsilon_r, cos_theta)
            >>> theta_d = jnp.rad2deg(theta)
            >>> plt.plot(theta_d, r_s, label=r"$r_s$")  # doctest: +SKIP
            >>> plt.plot(theta_d, r_p, label=r"$r_p$")  # doctest: +SKIP
            >>> plt.xlabel("Angle of incidence (°)")  # doctest: +SKIP
            >>> plt.ylabel("Power intensity coefficient")  # doctest: +SKIP
            >>> plt.legend()  # doctest: +SKIP
            >>> plt.tight_layout()  # doctest: +SKIP
    """
    sqrt_n_t = epsilon_r if mu_r is None else epsilon_r * mu_r

    n_t_cos_theta_t = jnp.sqrt(sqrt_n_t + cos_theta_i**2 - 1)
    n_t_squared_cos_theta_i = sqrt_n_t * cos_theta_i

    r_s = safe_divide(
        cos_theta_i - n_t_cos_theta_t,
        cos_theta_i + n_t_cos_theta_t,
    )
    r_p = safe_divide(
        n_t_squared_cos_theta_i - n_t_cos_theta_t,
        n_t_squared_cos_theta_i + n_t_cos_theta_t,
    )

    return r_s, r_p

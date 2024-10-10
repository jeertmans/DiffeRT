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
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Float, jaxtyped

from ..utils import safe_divide


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def reflection_coefficients(
    epsilon_r: Complex[Array, " *batch"],
    cos_theta_i: Float[Array, " *batch"],
    mu_r: Complex[Array, " *batch"] | None = None,
) -> tuple[Complex[Array, " *batch"], Complex[Array, " *batch"]]:
    r"""
    Compute the Fresnel reflection coefficients for air-to-dielectric interface.

    The Snell's law describes the relationship between the angles of incidence
    and refraction:

    .. math::
        n_i\sin\theta_i = n_t\sin\theta_t,

    where :math:`n` is the refraction index, :math:`theta` is the angle of between the ray path
    and the normal to the interface, and :math:`i` and :math:`t` indicate,
    respectively, the first (i.e., incidence) and the second (i.e., transmission)
    media.

    The s and p reflection coefficients are:

    .. math::
        \frac{r_s} = \frac{n_i\cos\theta_i - n_t\cos\theta_t}{n_i\cos\theta_i + n_t\cos\theta_t},

    and

    .. math::
        \frac{r_p} = \frac{n_t\cos\theta_i - n_i\cos\theta_t}{n_t\cos\theta_i + n_i\cos\theta_t}.

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
        cos_theta: The (cosine of the) angles of reflection.
        mu_r: The relative permeabilities. If not provided,
            a value of 1 is used.

    Returns:
        The reflection coefficients for s and p polarizations.
    """
    if mu_r is None:
        sqrt_n_t = epsilon_r
    else:
        sqrt_n_t = epsilon_r * mu_r

    n_t_cos_theta_t = jnp.sqrt(sqrt_n_t + cos_theta_i**2 - 1)
    n_t_squared_cos_theta_i = sqrt_n_t * cos_theta_i

    r_s = safe_divide(cos_theta_i - n_t_cos_theta_t, cos_theta_i + n_t_cos_theta_t)
    r_p = safe_divide(
        n_t_squared_cos_theta_i - n_t_cos_theta_t,
        n_t_squared_cos_theta_i + n_t_cos_theta_t,
    )

    return r_s, r_p

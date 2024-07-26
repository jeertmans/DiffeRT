"""
Fresnel coefficients for reflection and refraction.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Float, jaxtyped

from ..geometry.utils import normalize


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def reflection_coefficients(
    incident_rays: Float[Array, "*batch 3"],
    reflected_rays: Float[Array, "*batch 3"],
    normals: Float[Array, "*batch 3"],
) -> Complex[Array, " *batch 2 2"]:
    r"""
    Compute the Fresnel reflection coefficients.

    As detailed in :cite:`utd-mcnamara{eq. 3.199}`, the reflected field
    from a smooth surface can be expressed as:

    .. math::
        \boldsymbol{E}^r(P) = \boldsymbol{E}^r(Q_r) \sqrt{\frac{\pho_1^r\pho_2^r}{\left(\pho_1^r+s^r\right)\left(\pho_2^r+s^r\right)}} e^{-jks^r},

    where :math:`P` is the observation point and :math:`Q_r` is the reflection point on the surface. Hence, :math:`\boldsymbol{E}^r(Q_r)` can be expressed in terms of the incident field :math:`\boldsymbol{E}^i

    As detailed in :cite:`utd-mcnamara{p. 164}`, the integral can be expressed in
    terms of Fresnel integrals (:math:`C(z)` and :math:`S(z)`), so that:
    """
    # Normalize input rays
    incident_rays, _ = normalize(incident_rays)
    reflected_rays, _ = normalize(reflected_rays)

    # Compute the angle of incidence
    cos_theta_i = jnp.dot(incident_rays, normals)
    sin_theta_i = jnp.sqrt(1 - cos_theta_i**2)

    # Assume we're dealing with air-to-dielectric interface
    # You may need to adjust these values based on your specific scenario
    n1 = 1.0  # Refractive index of air
    n2 = 1.5  # Refractive index of the dielectric material

    # Compute the angle of transmission using Snell's law
    sin_theta_t = n1 * sin_theta_i / n2
    cos_theta_t = jnp.sqrt(1 - sin_theta_t**2)

    # Compute Fresnel coefficients for perpendicular polarization
    r_perp = (n1 * cos_theta_i - n2 * cos_theta_t) / (
        n1 * cos_theta_i + n2 * cos_theta_t
    )
    t_perp = (2 * n1 * cos_theta_i) / (n1 * cos_theta_i + n2 * cos_theta_t)

    # Compute Fresnel coefficients for parallel polarization
    r_para = (n2 * cos_theta_i - n1 * cos_theta_t) / (
        n2 * cos_theta_i + n1 * cos_theta_t
    )
    t_para = (2 * n1 * cos_theta_i) / (n2 * cos_theta_i + n1 * cos_theta_t)

    # Construct the Fresnel coefficient dyadic matrix
    r_matrix = jnp.array(
        [
            [r_perp, 0],
            [0, r_para],
        ],
    )

    t_matrix = jnp.array(
        [
            [t_perp, 0],
            [0, t_para],
        ],
    )

    return r_matrix, t_matrix

"""
Uniform Theory of Diffraction (UTD) utilities.

The foundamentals of UTD are described in :cite:`utd-mcnamara`.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=typechecker)
def F(z: Float[Array, " *batch"]) -> Complex[Array, " *batch"]:  # noqa: N802
    r"""
    Evaluate the transition function :cite:`utd-mcnamara{p. 184}` at the given points.

    The transition function is defined as follows:

    .. math::
        F(x) = 2j \sqrt{x} e^{j x} \int\limits_\sqrt{x}^\infty e^{-j u^2} \text{d}u,

    where :math:`j^2 = -1`.

    As detailed in :cite:`utd-mcnamara{p. 164}`, the integral can be expressed in
    terms of Fresnel integrals (:math:`C(x)` and :math:`S(x)`), so that:

    .. math::
        C(x) - j S(x) = \int\limits_\sqrt{x}^\infty e^{-j u^2} \text{d}u.

    Thus, the transition function can be rewritten as:

    .. math::
        2j \sqrt{z} e^{j z} \Big(\sqrt{\frac{\pi}{2}}\frac{1 - j}{2} - C(\sqrt{z}) + j S(\sqrt{z})\Big).

    With Fresnel integrals computed by :func:`jax.scipy.special.fresnel`.

    Args:
        z: The array of real points to evaluate.

    Returns:
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

    s, c = jsp.fresnel(sqrt_z / factor)
    return 2j * sqrt_z * jnp.exp(1j * z) * (factor * ((1 - 1j) / 2 - c + 1j * s))


@jax.jit
def rays_to_sin_angles(incident_rays, diffracted_rays, edge_vectors) -> None:
    """Compute the sin of angles..."""
    incident_rays, _ = normalize(incident_rays)
    diffracted_rays, _ = normalize(diffracted_rays)


@jax.jit
def diffraction_coefficients(
    incident_ray, diffracted_ray, edge_vector, k, n, r_prime, r, r0
):
    """
    Compute the diffraction coefficients based on the Uniform Theory of Diffraction.

    The implementation closely follows what is described
    in :cite:`utd-mcnamara{p. 268}`.

    Args:
        sin_beta_0: ...
        sin_beta: ...
        sin_phi:
    """
    # Ensure input vectors are normalized
    incident_ray = incident_ray / jnp.linalg.norm(incident_ray)
    diffracted_ray = diffracted_ray / jnp.linalg.norm(diffracted_ray)
    edge_vector = edge_vector / jnp.linalg.norm(edge_vector)

    # Compute relevant angles
    beta_0 = jnp.arccos(jnp.dot(incident_ray, edge_vector))
    beta = jnp.arccos(jnp.dot(diffracted_ray, edge_vector))
    phi = jnp.arccos(jnp.dot(-incident_ray, diffracted_ray))

    # Compute L parameters (distance parameters)
    L = r * jnp.sin(beta) ** 2 / (r + r_prime)
    L_prime = r_prime * jnp.sin(beta_0) ** 2 / (r + r_prime)

    # Compute the cotangent arguments
    cot_arg1 = (phi + (beta - beta_0)) / (2 * n)
    cot_arg2 = (phi - (beta - beta_0)) / (2 * n)
    cot_arg3 = (phi + (beta + beta_0)) / (2 * n)
    cot_arg4 = (phi - (beta + beta_0)) / (2 * n)

    # Define the cotangent function
    def cot(x):
        return 1.0 / jnp.tan(x)

    # Compute the a± coefficients
    1 + jnp.cos(2 * n * jnp.pi - (phi + beta - beta_0))
    1 + jnp.cos(2 * n * jnp.pi - (phi - beta + beta_0))

    # Compute the D_s and D_h functions
    def D_soft(L, cot_arg):
        return (
            -jnp.exp(-1j * jnp.pi / 4)
            / (2 * n * jnp.sqrt(2 * jnp.pi * k))
            * cot(cot_arg)
            * F(k * L * jnp.power(jnp.sin(cot_arg), 2))
        )

    def D_hard(L, cot_arg):
        return (
            -jnp.exp(-1j * jnp.pi / 4)
            / (2 * n * jnp.sqrt(2 * jnp.pi * k))
            * cot(cot_arg)
            * F(k * L * jnp.power(jnp.sin(cot_arg), 2))
        )

    # Compute the diffraction coefficients
    D_s = (
        D_soft(L, cot_arg1)
        + D_soft(L, cot_arg2)
        + D_soft(L_prime, cot_arg3)
        + D_soft(L_prime, cot_arg4)
    )

    D_h = (
        D_hard(L, cot_arg1)
        + D_hard(L, cot_arg2)
        + D_hard(L_prime, cot_arg3)
        + D_hard(L_prime, cot_arg4)
    )

    # Apply the Keller cone condition
    D_s = jnp.where(jnp.abs(jnp.sin(beta) - jnp.sin(beta_0)) < 1e-6, D_s, 0)
    D_h = jnp.where(jnp.abs(jnp.sin(beta) - jnp.sin(beta_0)) < 1e-6, D_h, 0)

    # Construct the dyadic diffraction coefficient matrix
    jnp.array([[D_s, 0, 0], [0, D_h, 0], [0, 0, 0]], dtype=jnp.complex64)

    # s_p

    d_12 = d_1 + d_2
    d_34 = d_3 + d_4

    d_s = d_12 - d_34
    d_h = d_12 + d_34

    return d_h, d_s

# ruff: noqa: N802, N806
from functools import partial
from typing import Any, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jaxtyping import Array, Complex, Float

from differt.utils import dot

# TODO: use ArrayLike instead of Array as inputs


@partial(jax.jit, inline=True)
def _cot(x: Float[Array, " *batch"]) -> Float[Array, " *batch"]:
    return 1 / jnp.tan(x)


@partial(jax.jit, inline=True)
def _sign(x: Float[Array, " *batch"]) -> Float[Array, " *batch"]:
    ones = jnp.ones_like(x)
    return jnp.where(x >= 0, ones, -ones)


@partial(jax.jit, inline=True, static_argnames=("mode"))
def _N(
    beta: Float[Array, " *#batch"], n: Float[Array, " *#batch"], mode: Literal["+", "-"]
) -> Float[Array, " *batch"]:
    if mode == "+":
        return jnp.round((beta + jnp.pi) / (2 * n * jnp.pi))
    return jnp.round((beta + jnp.pi) / (2 * n * jnp.pi))


@partial(jax.jit, inline=True, static_argnames=("mode"))
def _a(
    beta: Float[Array, " *#batch"], n: Float[Array, " *#batch"], mode: Literal["+", "-"]
) -> Float[Array, " *batch"]:
    N = _N(beta, n, mode)
    return 2.0 * jax.lax.integer_pow(jnp.cos(0.5 * (2 * n * jnp.pi * N - beta)), 2)


@overload
def L_i(
    s_d: Float[Array, " *#batch"],
    sin_2_beta_0: Float[Array, " *#batch"],
    rho_1_i: None = None,
    rho_2_i: None = None,
    rho_e_i: None = None,
    s_i: None = None,
) -> Float[Array, " *batch"]: ...


@overload
def L_i(
    s_d: Float[Array, " *#batch"],
    sin_2_beta_0: Float[Array, " *#batch"],
    rho_1_i: None = None,
    rho_2_i: None = None,
    rho_e_i: None = None,
    s_i: Float[Array, " *#batch"] | None = None,
) -> Float[Array, " *batch"]: ...


@overload
def L_i(
    s_d: Float[Array, " *#batch"],
    sin_2_beta_0: Float[Array, " *#batch"],
    rho_1_i: Float[Array, " *#batch"],
    rho_2_i: Float[Array, " *#batch"],
    rho_e_i: Float[Array, " *#batch"],
    s_i: None = None,
) -> Float[Array, " *batch"]: ...


@eqx.filter_jit
def L_i(  # noqa: PLR0917
    s_d: Float[Array, " *#batch"],
    sin_2_beta_0: Float[Array, " *#batch"],
    rho_1_i: Float[Array, " *#batch"] | None = None,
    rho_2_i: Float[Array, " *#batch"] | None = None,
    rho_e_i: Float[Array, " *#batch"] | None = None,
    s_i: Float[Array, " *#batch"] | None = None,
) -> Float[Array, " *batch"]:
    r"""
    Compute the distance parameter associated with the incident shadow boundaries.

    .. note::

        This function can also be used to compute the distance parameters
        associated with the reflection shadow boundaries for the o- and n-faces,
        by passing the corresponding radii of curvature, see
        :cite:`utd-mcnamara{eq. 6.28, p. 270}`.

    Its general expression is given by :cite:`utd-mcnamara{eq. 6.25, p. 270}`:

    .. math::
        L_i = \frac{(\rho_e^i + s)\rho_1^i\rho_2^i}{\rho_e^i(\rho_1^i + s)(\rho_2^i + s)}\sin^2\beta_0,

    where :math:`s^d` is the distance from the point of diffraction (:math:`Q_d`) to the observer
    point (:math:`P`),
    :math:`\rho_1^i` is the principal radius of curvature of the incident wavewront at :math:`Q_d`
    in the plane of incidence,
    :math:`\rho_2^i` is the principal radius of curvature of the incident wavewront at :math:`Q_d`
    in the place transverse to the plane of incidence,
    :math:`\rho_e^i` is radius of curvature of the incident wavefront in the edge-fixed
    plane of incidence., and :math:`\beta_0` is the angle of diffraction.

    By default, when :math:`\rho_1^i`, :math:`\rho_e^i`, and :math:`\rho_2^i` are not provided,
    a plane wave incidence is assumed and the expression simplifies to
    :cite:`utd-mcnamara{eq. 6.27, p. 270}`:

    .. math::
        L_i = s^d\sin^2\beta_0.

    For spherical wavefront, you can pass :math:`s^i` (**s\_i**), the radius of curvature of
    the spherical wavefront, where :math:`s^i = \rho_1^i = \rho_2^2 = \rho_e^i`,
    and the expression will be simplified to
    :cite:`utd-mcnamara{eq. 6.26, p. 270}`:

    .. math::
        L_i = \frac{s^ds^i}{s^d + s^i}\sin^2\beta_0.

    Args:
        s_d: The distance from :math:`Q_d` to :math:`P`.
        sin_2_beta_0: The squared sine of the angle of diffraction.
        rho_1_i: The principal radius of curvature of the incident wavefront
            in the plane of incidence.
        rho_2_i: The principal radius of curvature of the incident wavefront
            in the plane transverse to the plane of incidence.
        rho_e_i: The radius of curvature of the incident wavefront in the edge-fixed
            plane of incidence.
        s_i: The radius of curvature of the incident spherical wavefront.

            If this is set, other radius parameters must be set to 'None'.

    Returns:
        The values of the distance parameter :math:`L_i`.

    Raises:
        ValueError: If 's_i' was provided along at least one of the other radius parameters,
            or if one or the three 'rho' parameters was not provided.
    """
    radii = (rho_1_i, rho_2_i, rho_e_i)
    all_none = all(x is None for x in radii)
    all_set = all(x is not None for x in radii)
    if s_i is not None and any(x is not None for x in radii):
        msg = "If 's_i' is provided, then 'rho_1_i', 'rho_2_i', and 'rho_e_i' must be left to 'None'."
        raise ValueError(msg)
    if (not all_none) and (not all_set):
        msg = "All three of 'rho_1_i', 'rho_2_i', and 'rho_e_i' must be provided, or left to 'None'."
        raise ValueError(msg)

    if s_i is not None:
        return (s_d * s_i) * sin_2_beta_0 / (s_d + s_i)
    if all_none:
        return s_d * sin_2_beta_0
    return (
        (s_d * (rho_e_i + s_d) * rho_1_i * rho_2_i)
        / (rho_e_i * (rho_1_i + s_d) * (rho_2_i + s_d))
    ) * sin_2_beta_0


@jax.jit
def F(z: Float[Array, " *batch"]) -> Complex[Array, " *batch"]:
    r"""
    Evaluate the transition function at the given points.

    The transition function is defined as follows :cite:`utd-mcnamara{eq. 4.72, p. 184}`:

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

    With Fresnel integrals computed by :data:`jax.scipy.special.fresnel`.

    Args:
        z: The array of real points to evaluate.

    Returns:
        The values of the transition function at the given point.

    Examples:
        .. plot::

            The following example reproduces the same plot as in
            :cite:`utd-mcnamara{fig. 4.16, p. 185}`.

            >>> from differt.em import F
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
def diffraction_coefficients(
    *_args: Any,
) -> None:
    """
    Compute the diffraction coefficients based on the Uniform Theory of Diffraction.

    Warning:
        This function is not yet implemented, as we are still thinking of the
        best API for it. If you want to get involved in the implementation of UTD coefficients,
        please reach out to us on GitHub!

    The implementation closely follows what is described
    in :cite:`utd-mcnamara{p. 268-273}`.

    Unlike :func:`fresnel_coefficients<differt.em.fresnel_coefficients>`, diffraction
    coefficients depend on the radii of curvature of the incident wave.

    Args:
        sin_beta_0: ...
        sin_beta: ...
        sin_phi: ...
        rho_1_i: ...
        rho_1_i: ...
        rho_e_i: ...

    Returns:
        The soft and hard diffraction coefficients.

    Raises:
        NotImplementedError: The function is not yet implemented.
    """
    # ruff: noqa: ERA001, F821, F841
    raise NotImplementedError

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

    phi_i = jnp.pi - (jnp.pi - jnp.arccos(dot(-s_t_i, t_o))) * _sign(dot(-s_t_i, n_o))
    phi_d = jnp.pi - (jnp.pi - jnp.arccos(dot(+s_t_d, t_o))) * _sign(dot(+s_d_i, n_o))

    # Compute the angle differences
    phi_1 = phi_d - phi_i
    phi_2 = phi_d + phi_i

    # Compute the diffraction coefficients (without common mul. factor)
    D_1 = _cot((jnp.pi + phi_1) / (2 * n)) * F(k * L_i * _a(phi_1, "+"))
    D_2 = _cot((jnp.pi - phi_1) / (2 * n)) * F(k * L_i * _a(phi_1, "-"))
    D_3 = _cot((jnp.pi + phi_2) / (2 * n)) * F(k * L_r_n * _a(phi_2, "+"))
    D_4 = _cot((jnp.pi - phi_2) / (2 * n)) * F(k * L_r_o * _a(phi_2, "-"))

    factor = -jnp.exp(-1j * jnp.pi / 4) / (
        2 * n * jnp.sqrt(2 * jnp.pi * k) * sin_beta_0
    )

    # Apply the Keller cone condition
    # D_s = jnp.where(jnp.abs(jnp.sin(beta) - jnp.sin(beta_0)) < 1e-6, D_s, 0)
    # D_h = jnp.where(jnp.abs(jnp.sin(beta) - jnp.sin(beta_0)) < 1e-6, D_h, 0)

    # TODO: below are assuming perfectly conducting surfaces

    D_12 = D_1 + D_2
    D_34 = D_3 + D_4

    D_s = (D_12 - D_34) * factor
    D_h = (D_12 + D_34) * factor

    return D_s, D_h

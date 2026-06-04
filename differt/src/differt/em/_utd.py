# ruff: noqa: N802, N803, N806
# type: ignore  # noqa: PGH003
from typing import Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, Complex, Float

from differt.em._fresnel import reflection_coefficients

# TODO: use ArrayLike instead of Array as inputs


@jax.jit(inline=True)
def _cot(x: Float[Array, " *batch"]) -> Float[Array, " *batch"]:
    return 1 / jnp.tan(x)


@jax.jit(inline=True)
def _sign(x: Float[Array, " *batch"]) -> Float[Array, " *batch"]:
    ones = jnp.ones_like(x)
    return jnp.where(x >= 0, ones, -ones)


@jax.jit(inline=True, static_argnames=("mode"))
def _N(
    beta: Float[Array, " *#batch"], n: Float[Array, " *#batch"], mode: Literal["+", "-"]
) -> Float[Array, " *batch"]:
    if mode == "+":
        return jnp.round((beta + jnp.pi) / (2 * n * jnp.pi))
    return jnp.round((beta - jnp.pi) / (2 * n * jnp.pi))


@jax.jit(inline=True, static_argnames=("mode"))
def _a(
    beta: Float[Array, " *#batch"], n: Float[Array, " *#batch"], mode: Literal["+", "-"]
) -> Float[Array, " *batch"]:
    N = _N(beta, n, mode)
    return 2.0 * jax.lax.integer_pow(jnp.cos(0.5 * (2 * n * jnp.pi * N - beta)), 2)


@overload
def L_i(
    s_d: Float[Array, " *#batch"],
    sin_2_beta_0: Float[Array, " *#batch"],
    rho_1_i: None = ...,
    rho_2_i: None = ...,
    rho_e_i: None = ...,
    s_i: None = ...,
) -> Float[Array, " *batch"]: ...


@overload
def L_i(
    s_d: Float[Array, " *#batch"],
    sin_2_beta_0: Float[Array, " *#batch"],
    rho_1_i: None = ...,
    rho_2_i: None = ...,
    rho_e_i: None = ...,
    s_i: Float[Array, " *#batch"] | None = ...,
) -> Float[Array, " *batch"]: ...


@overload
def L_i(
    s_d: Float[Array, " *#batch"],
    sin_2_beta_0: Float[Array, " *#batch"],
    rho_1_i: Float[Array, " *#batch"],
    rho_2_i: Float[Array, " *#batch"],
    rho_e_i: Float[Array, " *#batch"],
    s_i: None = ...,
) -> Float[Array, " *batch"]: ...


@eqx.filter_jit
def L_i(
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
    :math:`\rho_1^i` is the principal radius of curvature of the incident wavefront at :math:`Q_d`
    in the plane of incidence,
    :math:`\rho_2^i` is the principal radius of curvature of the incident wavefront at :math:`Q_d`
    in the plane transverse to the plane of incidence,
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


@jax.jit(inline=True, static_argnames=("mode"))
def _cot_times_F(
    beta: Float[Array, " *#batch"],
    n: Float[Array, " *#batch"],
    k: Float[Array, " *#batch"],
    L_val: Float[Array, " *#batch"],
    mode: Literal["+", "-"],
) -> Complex[Array, " *batch"]:
    mode_sign = 1.0 if mode == "+" else -1.0
    N = _N(beta, n, mode)
    beta_boundary = 2 * n * jnp.pi * N - mode_sign * jnp.pi
    delta = beta - beta_boundary

    threshold = 1e-5
    is_near_boundary = jnp.abs(delta) < threshold

    # Limit value as delta -> 0
    limit_val = (
        mode_sign
        * _sign(delta)
        * (2.0 * n)
        * (1.0 + 1j)
        * jnp.sqrt(jnp.pi * k * L_val / 2.0)
    )

    # Safe delta to prevent division by zero/NaN inside unselected paths in jnp.where
    delta_safe = jnp.where(is_near_boundary, threshold, delta)
    x_safe = N * jnp.pi + mode_sign * delta_safe / (2 * n)
    a_safe = 2.0 * jnp.sin(delta_safe / 2.0) ** 2
    z_safe = k * L_val * a_safe

    prod_safe = _cot(x_safe) * F(z_safe)
    return jnp.where(is_near_boundary, limit_val, prod_safe)


@jax.jit
def diffraction_coefficients(
    wavenumber: Float[ArrayLike, " *batch"],
    n: Float[ArrayLike, " *batch"],
    phi_i: Float[ArrayLike, " *batch"],
    phi_d: Float[ArrayLike, " *batch"],
    L_i: Float[ArrayLike, " *batch"],
    L_r_n: Float[ArrayLike, " *batch"] | None = None,
    L_r_o: Float[ArrayLike, " *batch"] | None = None,
    sin_beta_0: Float[ArrayLike, " *batch"] | None = None,
    n_r_o: Complex[ArrayLike, " *batch"] | None = None,
    n_r_n: Complex[ArrayLike, " *batch"] | None = None,
) -> tuple[Complex[Array, " *batch"], Complex[Array, " *batch"]]:
    r"""
    Compute the diffraction coefficients based on the Uniform Theory of Diffraction.

    The implementation closely follows what is described
    in :cite:`utd-mcnamara{p. 268-273}`.

    Unlike :func:`fresnel_coefficients<differt.em.fresnel_coefficients>`, diffraction
    coefficients depend on the radii of curvature of the incident wave and the wedge geometry.

    Args:
        wavenumber: The wavenumber :math:`k`.
        n: The wedge parameter :math:`n` (exterior wedge angle is :math:`n \pi`).
        phi_i: The incident angle :math:`\phi_i` (or :math:`\phi'`) in radians,
            measured from the o-face.
        phi_d: The diffracted angle :math:`\phi_d` (or :math:`\phi`) in radians,
            measured from the o-face.
        L_i: The distance parameter associated with the incident shadow boundary.
        L_r_n: The distance parameter associated with the reflection shadow boundary
            for the n-face. Defaults to ``L_i``.
        L_r_o: The distance parameter associated with the reflection shadow boundary
            for the o-face. Defaults to ``L_i``.
        sin_beta_0: The sine of the angle of incidence on the edge, :math:`\sin \beta_0`.
            Defaults to ``1.0`` (broadside incidence).
        n_r_o: The relative refractive index of the o-face material.
            If ``None``, the o-face is assumed to be a Perfect Electric Conductor (PEC).
        n_r_n: The relative refractive index of the n-face material.
            If ``None``, the n-face is assumed to be a Perfect Electric Conductor (PEC).

    Returns:
        A tuple containing the soft and hard diffraction coefficients :math:`(D_s, D_h)`.
    """
    k = jnp.maximum(jnp.asarray(wavenumber), 1e-12)
    n_arr = jnp.asarray(n)
    phi_i_arr = jnp.asarray(phi_i)
    phi_d_arr = jnp.asarray(phi_d)
    L_i_arr = jnp.maximum(jnp.asarray(L_i), 1e-12)

    L_r_n_arr = L_i_arr if L_r_n is None else jnp.maximum(jnp.asarray(L_r_n), 1e-12)
    L_r_o_arr = L_i_arr if L_r_o is None else jnp.maximum(jnp.asarray(L_r_o), 1e-12)

    if sin_beta_0 is None:
        sin_beta_0_safe = jnp.ones_like(k)
    else:
        sin_beta_0_safe = jnp.maximum(jnp.asarray(sin_beta_0), 1e-12)

    # Angle differences
    phi_1 = phi_d_arr - phi_i_arr
    phi_2 = phi_d_arr + phi_i_arr

    # Compute the four terms using the NaN-safe helper
    D_1 = _cot_times_F(phi_1, n_arr, k, L_i_arr, "+")
    D_2 = _cot_times_F(phi_1, n_arr, k, L_i_arr, "-")
    D_3 = _cot_times_F(phi_2, n_arr, k, L_r_n_arr, "+")
    D_4 = _cot_times_F(phi_2, n_arr, k, L_r_o_arr, "-")

    # Common multiplicative factor
    factor = -jnp.exp(-1j * jnp.pi / 4) / (
        2.0 * n_arr * jnp.sqrt(2.0 * jnp.pi * k) * sin_beta_0_safe
    )

    # Calculate reflection coefficients for o-face and n-face (Luebbers' model)
    dtype = jnp.result_type(k)
    complex_dtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64

    if n_r_o is None:
        r_s_o = jnp.full_like(k, -1.0, dtype=complex_dtype)
        r_p_o = jnp.full_like(k, 1.0, dtype=complex_dtype)
    else:
        cos_theta_o = jnp.sin(phi_i_arr)
        r_s_o, r_p_o = reflection_coefficients(n_r_o, cos_theta_o)

    if n_r_n is None:
        r_s_n = jnp.full_like(k, -1.0, dtype=complex_dtype)
        r_p_n = jnp.full_like(k, 1.0, dtype=complex_dtype)
    else:
        cos_theta_n = jnp.sin(n_arr * jnp.pi - phi_i_arr)
        r_s_n, r_p_n = reflection_coefficients(n_r_n, cos_theta_n)

    D_s = (D_1 + D_2 + r_s_n * D_3 + r_s_o * D_4) * factor
    D_h = (D_1 + D_2 + r_p_n * D_3 + r_p_o * D_4) * factor

    return D_s, D_h

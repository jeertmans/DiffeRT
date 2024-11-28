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
    r"""
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

    Examples:
        .. plot::
           :context: reset

           The following example show how to compute interference
           patterns from line of sight and reflection on a glass
           ground.

           >>> from differt.em import reflection_coefficients
           >>> from differt.em import Dipole
           >>> from differt.em import pointing_vector
           >>> from differt.geometry import assemble_paths, normalize, path_lengths
           >>> from differt.rt import image_method
           >>> from differt.utils import dot

           The first step is to define the antenna and the geometry of the scene.
           Here, we place a dipole antenna above the origin, and generate a
           ``num_positions`` number of positions along the horizontal line,
           where we will evaluate the EM fields.

           >>> tx_position = jnp.array([0.0, 2.0, 0.0])
           >>> rx_position = jnp.array([0.0, 2.0, 0.0])
           >>> num_positions = 1000
           >>> x = jnp.logspace(0, 3, num_positions)  # From close to very far
           >>> rx_positions = (
           ...     jnp.tile(rx_position, (num_positions, 1)).at[..., 0].add(x)
           ... )
           >>> ant = Dipole(2.4e9)  # 2.4 GHz
           >>> plt.xscale("symlog", linthresh=1e-1)  # doctest: +SKIP
           >>> plt.plot(
           ...     [tx_position[0]],
           ...     [tx_position[1]],
           ...     "o",
           ...     label="TX",
           ... )  # doctest: +SKIP
           >>> plt.plot(
           ...     rx_positions[::50, 0],
           ...     rx_positions[::50, 1],
           ...     "o",
           ...     label="RXs",
           ... )  # doctest: +SKIP
           >>> plt.axhline(color="k", label="Ground")  # doctest: +SKIP
           >>> plt.xlabel("x-axis (m)")  # doctest: +SKIP
           >>> plt.ylabel("y-axis (m)")  # doctest: +SKIP
           >>> plt.legend()  # doctest: +SKIP
           >>> plt.tight_layout()  # doctest: +SKIP

        .. plot::
           :context: close-figs

           Next, we compute the EM fields from the direct (line-of-sight) path.

           >>> e_los, b_los = ant.fields(rx_positions - tx_position)
           >>> p_los = jnp.linalg.norm(pointing_vector(e_los, b_los), axis=-1)
           >>> plt.semilogx(
           ...     x,
           ...     10 * jnp.log10(p_los / ant.average_power),
           ...     label=r"$P_\text{los}$",
           ... )  # doctest: +SKIP

           After, the :func:`image_method<differt.rt.image_method>`
           function is used to compute the reflection points. We then compute the EM fields
           at those points, and use the Fresnel reflection coefficients to compute the
           reflected fields.

           >>> ground_vertex = jnp.array([0.0, 0.0, 0.0])
           >>> ground_normal = jnp.array([0.0, 1.0, 0.0])
           >>> reflection_points = image_method(
           ...     tx_position,
           ...     rx_positions,
           ...     ground_vertex[None, ...],
           ...     ground_normal[None, ...],
           ... ).squeeze(axis=-2)  # Squeeze because only one reflection
           >>> e_at_rp, b_at_rp = ant.fields(reflection_points - tx_position)
           >>> incident_vectors, s = normalize(
           ...     reflection_points - tx_position, keepdims=True
           ... )
           >>> cos_theta = dot(ground_normal, -incident_vectors)
           >>> n_r = 1.5  # Glass
           >>> r_s, r_p = reflection_coefficients(n_r, cos_theta)
           >>> e_s = normalize(
           ...     jnp.cross(
           ...         incident_vectors, jnp.cross(ground_normal, incident_vectors)
           ...     )
           ... )[0]
           >>> e_p = jnp.cross(incident_vectors, e_s)
           >>> e_refl = r_s[:, None] * e_s * e_at_rp + r_p[:, None] * e_p * e_at_rp
           >>> b_refl = r_s[:, None] * e_s * b_at_rp + r_p[:, None] * e_p * b_at_rp

           .. important::

              Reflection coefficients are returned based on s and p directions.
              As a result, we need to first determine those local directions, and
              apply the corresponding reflection coefficients to the projection
              of the fields onto those directions
              :cite:`utd-mcnamara{eq. 3.3-3.8 and 3.39, p. 70 and 77}`.

              After reflection, the s direction stays the same, and the p direction is reversed.

           Finally, we apply the spreading factor and phase shift due to the propagation
           from the reflection points to the receiver :cite:`utd-mcnamara{eq. 3.1, p. 63}`.

           >>> reflected_vectors, s_r = normalize(
           ...     reflection_points - rx_positions, keepdims=True
           ... )
           >>> spreading_factor = s / (
           ...     s + s_r
           ... )  # We assume that the radii of curvature are equal to 's'
           >>> phase_shift = jnp.exp(1j * s_r * ant.wavenumber)
           >>> e_refl *= spreading_factor * phase_shift
           >>> b_refl *= spreading_factor * phase_shift
           >>> p_refl = jnp.linalg.norm(pointing_vector(e_refl, b_refl), axis=-1)
           >>> plt.semilogx(
           ...     x,
           ...     10 * jnp.log10(p_refl / ant.average_power),
           ...     "--",
           ...     label=r"$P_\text{reflection}$",
           ... )  # doctest: +SKIP

           We also plot the total field, to better observe the interference pattern.

           >>> e_tot = e_los + e_refl
           >>> b_tot = b_los + b_refl
           >>> p_tot = jnp.linalg.norm(pointing_vector(e_tot, b_tot), axis=-1)
           >>> plt.semilogx(
           ...     x,
           ...     10 * jnp.log10(p_tot / ant.average_power),
           ...     "-.",
           ...     label=r"$P_\text{total}$",
           ... )  # doctest: +SKIP
           >>> plt.xlabel("Distance to transmitter on x-axis (m)")  # doctest: +SKIP
           >>> plt.ylabel("Loss (dB)")  # doctest: +SKIP
           >>> plt.legend()  # doctest: +SKIP
           >>> plt.tight_layout()  # doctest: +SKIP

        From the above figure, it is clear that the ground-reflection creates an interference
        pattern in the received power. Moreover, we can clearly observe the Brewster angle
        at a distance of 6 m. This can verified by computing the Brewster angle from the
        relative refractive index, and matching it to the corresponding distance.

        >>> brewster_angle = jnp.arctan(n_r)
        >>> print(f"Brewster angle: {jnp.rad2deg(brewster_angle):.1f}°")
        Brewster angle: 56.3°
        >>> cos_distance = jnp.abs(jnp.cos(brewster_angle) - cos_theta)
        >>> distance = x[jnp.argmin(cos_distance)]
        >>> print(f"Corresponding distance: {distance:.1f} m")
        Corresponding distance: 6.0 m
    """
    # TODO: check in the above example the s-p polarization directions are correct
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

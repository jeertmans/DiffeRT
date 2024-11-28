r"""
Electromagnetic (EM) fields utilities.

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
   >>> rx_positions = jnp.tile(rx_position, (num_positions, 1)).at[..., 0].add(x)
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
   >>> plt.plot(
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
   >>> incident_vectors, s = normalize(reflection_points - tx_position, keepdims=True)
   >>> cos_theta = dot(ground_normal, -incident_vectors)
   >>> n_r = 1.5  # Glass
   >>> r_s, r_p = reflection_coefficients(n_r, cos_theta)
   >>> e_s = normalize(
   ...     jnp.cross(incident_vectors, jnp.cross(ground_normal, incident_vectors))
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

__all__ = (
    "Antenna",
    "Dipole",
    "F",
    "ShortDipole",
    "c",
    "diffraction_coefficients",
    "epsilon_0",
    "fresnel_coefficients",
    "lengths_to_delays",
    "mu_0",
    "path_delays",
    "pointing_vector",
    "reflection_coefficients",
    "refraction_coefficients",
    "refractive_indices",
    "z_0",
)

from ._antenna import Antenna, Dipole, ShortDipole, pointing_vector
from ._fresnel import (
    fresnel_coefficients,
    reflection_coefficients,
    refraction_coefficients,
    refractive_indices,
)
from ._utd import F, diffraction_coefficients
from ._utils import lengths_to_delays, path_delays
from .constants import c, epsilon_0, mu_0, z_0

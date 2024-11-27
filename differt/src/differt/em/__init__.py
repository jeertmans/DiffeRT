r"""
Electromagnetic fields utilities.

.. plot::

   The following example show how to compute interference
   patterns from line of sight and reflection on a metal
   ground.

   >>> from differt.em import reflection_coefficients
   >>> from differt.em import Dipole
   >>> from differt.em import pointing_vector
   >>> from differt.geometry import assemble_paths, normalize, path_lengths
   >>> from differt.rt import image_method
   >>>
   >>> tx_position = jnp.array([0.0, 2.0, 0.0])
   >>> rx_position = jnp.array([0.0, 1.0, 0.0])
   >>> n = 1000
   >>> x = jnp.linspace(1, 10, n)
   >>> rx_positions = jnp.tile(rx_position, (n, 1)).at[..., 0].add(x)
   >>> ant = Dipole(1e9)  # 1 GHz
   >>> e_los, b_los = ant.fields(rx_positions - tx_position)
   >>> p_los = jnp.linalg.norm(pointing_vector(e_los, b_los), axis=-1)
   >>> plt.plot(
   ...     x,
   ...     20 * jnp.log10(p_los / ant.average_power),
   ...     label=r"$P_\text{los}$",
   ... )  # doctest: +SKIP
   >>>
   >>> ground_vertex = jnp.array([0.0, 0.0, 0.0])
   >>> ground_normal = jnp.array([0.0, 1.0, 0.0])
   >>> reflection_points = image_method(
   ...     tx_position,
   ...     rx_positions,
   ...     ground_vertex[None, ...], ground_normal[None, ...],
   ... )
   >>> e_at_rp, b_at_rp = ant.fields(reflection_points - tx_position).squeeze()
   >>> incident_vectors = normalize(reflection_points - tx_position)[0]
   >>> cos_theta = jnp.sum(ground_normal * -incident_vectors, axis=-1)
   >>> epsilon_r = 1.0
   >>> r_s, r_p = reflection_coefficients(epsilon_r, cos_theta)
   >>> e_refl = r_s[:, None] * cos_theta[:, None] * e_at_rp
   >>> r_s.shape, cos_theta.shape, e_at_rp.shape, e_refl.shape
   >>> b_refl = r_s[:, None] * cos_theta[:, None] * b_at_rp
   >>> s = jnp.linalg.norm(reflection_points - rx_positions, axis=-1, keepdims=True)
   >>> spreading_factor = 1 / s
   >>> phase_shift = jnp.exp(1j * s * ant.wavenumber)
   >>> e_refl *= spreading_factor * phase_shift
   >>> b_refl *= spreading_factor * phase_shift
   >>> p_refl = jnp.linalg.norm(pointing_vector(e_refl, b_refl), axis=-1)
   >>> plt.plot(
   ...     x,
   ...     20 * jnp.log10(p_refl / ant.average_power),
   ...     label=r"$P_\text{reflection}$",
   ... )  # doctest: +SKIP
   >>> plt.xlabel("Distance to transmitter on x-axis")  # doctest: +SKIP
   >>> plt.ylabel("Loss (dB)")  # doctest: +SKIP
   >>> plt.legend()  # doctest: +SKIP
   >>> plt.tight_layout()  # doctest: +SKIP
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

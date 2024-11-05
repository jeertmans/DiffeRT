``differt.em`` module
=====================

.. currentmodule:: differt.em

.. automodule:: differt.em

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
   >>> e_at_rp, b_at_rp = ant.fields(reflection_points - tx_position)
   >>> incident_vectors = normalize(reflection_points - tx_position)[0]
   >>> cos_theta = jnp.sum(ground_normal * -incident_vectors, axis=-1)
   >>> epsilon_r = 1.0
   >>> r_s, r_p = reflection_coefficients(epsilon_r, cos_theta)
   >>> # theta_d = jnp.rad2deg(theta)
   >>> plt.xlabel("Distance to transmitter on x-axis")  # doctest: +SKIP
   >>> plt.ylabel("Loss (dB)")  # doctest: +SKIP
   >>> plt.legend()  # doctest: +SKIP
   >>> plt.tight_layout()  # doctest: +SKIP


.. rubric:: Constants

Electrical constants (re-exported to this module).

.. currentmodule:: differt.em.constants

.. autosummary::
   :toctree: _autosummary

   c
   epsilon_0
   mu_0
   z_0

.. currentmodule:: differt.em

.. rubric:: Fields coefficients

Fresnel and diffraction coefficients, as described by the Geometrical Optics (GO)
and the Uniform Theory of Diffraction (UTD).

As detailed in :cite:`utd-mcnamara{eq. 3.199}`, the GO reflected field
from a smooth conducting surface can be expressed as:

.. math::
    \boldsymbol{E}^r(P) = \boldsymbol{E}^r(Q_r) \sqrt{\frac{\rho_1^r\rho_2^r}{\left(\rho_1^r+s^r\right)\left(\rho_2^r+s^r\right)}} e^{-jks^r},

where :math:`P` is the observation point and :math:`Q_r` is the reflection point on the surface, :math:`\rho_1^r` and :math:`\rho_2^r` are the principal radii of curvature at :math:`Q_r` of the reflected wavefront, :math:`k` is the wavenumber, and :math:`s_r` is the distance between :math:`Q_r` and :math:`P`. Moreover, :math:`\boldsymbol{E}^r(Q_r)` can be expressed in terms of the incident field :math:`\boldsymbol{E}^i`:

.. math::
    \boldsymbol{E}^r(Q_r) = \boldsymbol{E}^r(Q_r) \cdot R

where :math:`\boldsymbol{R}` is the dyadic matrix with the reflection coefficients.

The foundamentals of UTD are described are also described in :cite:`utd-mcnamara`,
where Chapter 6 (p. 263) covers three-dimension wedge diffraction.

.. autosummary::
   :toctree: _autosummary

   diffraction_coefficients
   fresnel_coefficients
   reflection_coefficients
   refraction_coefficients
   refractive_indices

.. rubric:: Antennas

The following antenna classes are defined to work in vacuum.
If you want to use those classes in another media, you can do so
by multiplying the output fields by relative permeabilities and permittivities,
when relevant.

.. autosummary::
   :toctree: _autosummary

   Antenna
   Dipole
   ShortDipole

.. rubric:: Utilities

Utility functions, mostly used internally for computing EM fields.

.. autosummary::
   :toctree: _autosummary

   lengths_to_delays
   path_delays
   pointing_vector

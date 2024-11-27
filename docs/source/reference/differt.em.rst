``differt.em`` module
=====================

.. currentmodule:: differt.em

.. automodule:: differt.em


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

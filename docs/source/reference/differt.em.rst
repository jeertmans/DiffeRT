``differt.em`` module
=====================

.. currentmodule:: differt.em

.. automodule:: differt.em

.. rubric:: Constants

Electrical constants used for EM fields computation.

..
   _See https://github.com/sphinx-doc/sphinx/issues/6495 to understand why
   we need to change the current module.

.. currentmodule:: differt.em._constants

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

As detailed in :cite:`utd-mcnamara{eq. 3.199, p. 131}`, the GO reflected field
from a smooth conducting surface can be expressed as:

.. math::
    \boldsymbol{E}^r(P) = \boldsymbol{E}^r(Q_r) \sqrt{\frac{\rho_1^r\rho_2^r}{\left(\rho_1^r+s^r\right)\left(\rho_2^r+s^r\right)}} e^{-jks^r},

where :math:`P` is the observation point and :math:`Q_r` is the reflection point on the surface, :math:`\rho_1^r` and :math:`\rho_2^r` are the principal radii of curvature at :math:`Q_r` of the reflected wavefront, :math:`k` is the wavenumber, and :math:`s_r` is the distance between :math:`Q_r` and :math:`P`. Moreover, :math:`\boldsymbol{E}^r(Q_r)` can be expressed in terms of the incident field :math:`\boldsymbol{E}^i`:

.. math::
    \boldsymbol{E}^r(Q_r) = \boldsymbol{E}^i(Q_r) \cdot \boldsymbol{R}

where :math:`\boldsymbol{R}` is the dyadic matrix with the reflection coefficients.

The foundamentals of UTD are described are also described in :cite:`utd-mcnamara`,
where Chapter 6 (p. 263) covers three-dimension wedge diffraction. A similar expression
can be obtained to express the diffraction field in function of the incident field
:cite:`utd-mcnamara{eq. 6.13, p. 268}`:

.. math::
    \boldsymbol{E}^d(P) = \boldsymbol{E}^d(Q_d) \sqrt{\frac{\rho^d}{\left(\rho_1^d+s^r\right)\left(\rho_2^r+s^r\right)}} e^{-jks^d},

where :math:`P` is the observation point and :math:`Q_d` is the diffraction point on the edge, :math:`\rho^d` is the edge caustic distance, :math:`k` is the wavenumber, and :math:`s_d` is the distance between :math:`Q_r` and :math:`P`. Moreover, :math:`\boldsymbol{E}^d(Q_d)` can be expressed in terms of the incident field :math:`\boldsymbol{E}^i`:

.. math::
    \boldsymbol{E}^d(Q_d) = \boldsymbol{E}^i(Q_d) \cdot \boldsymbol{D}

where :math:`\boldsymbol{D}` is the dyadic matrix with the diffraction coefficients.

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

.. rubric:: Materials

We provide a basic class to represent radio materials,
and a mapping containing some common materials (e.g., ITU-R materials).

.. currentmodule:: differt.em._material

.. autosummary::
   :toctree: _autosummary

   Material
   materials

.. currentmodule:: differt.em

Types of interaction (reflection, diffraction, etc.) within a path
are identified by different numbers, which are listed in an enum class.

.. autosummary::
   :toctree: _autosummary

   InteractionType

.. rubric:: Utilities

Utility functions, mostly used internally for computing EM fields.

.. autosummary::
   :toctree: _autosummary

   lengths_to_delays
   path_delays
   pointing_vector
   sp_directions
   F
   L_i

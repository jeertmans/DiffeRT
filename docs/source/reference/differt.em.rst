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

The fundamentals of UTD are also described in :cite:`utd-mcnamara`,
where Chapter 6 (p. 263) covers three-dimensional wedge diffraction. A similar expression
can be obtained to express the diffraction field as a function of the incident field
:cite:`utd-mcnamara{eq. 6.13, p. 268}`:

.. math::
    \boldsymbol{E}^d(P) = \boldsymbol{E}^d(Q_d) \sqrt{\frac{\rho^d}{s^d\left(\rho^d+s^d\right)}} e^{-jks^d},

where :math:`P` is the observation point and :math:`Q_d` is the diffraction point on the edge, :math:`\rho^d` is the edge caustic distance, :math:`k` is the wavenumber, and :math:`s^d` is the distance between :math:`Q_d` and :math:`P`. Moreover, :math:`\boldsymbol{E}^d(Q_d)` can be expressed in terms of the incident field :math:`\boldsymbol{E}^i`:

.. math::
    \boldsymbol{E}^d(Q_d) = \boldsymbol{E}^i(Q_d) \cdot \boldsymbol{D}

where :math:`\boldsymbol{D}` is the dyadic matrix with the diffraction coefficients.

.. autosummary::
   :toctree: _autosummary

   fresnel_coefficients
   reflection_coefficients
   refraction_coefficients
   refractive_index
   slab_coefficients
   diffraction_coefficients
   F
   L_i

.. rubric:: Antennas

The following antenna classes are defined to work in vacuum.
If you want to use those classes in another medium, you can do so
by multiplying the output fields by relative permeabilities and permittivities,
when relevant.

Each :class:`AbstractAntenna` implements :meth:`AbstractAntenna.wavefront_radii`, which
tells :class:`GeometricFieldSolver<differt.em.GeometricFieldSolver>` the
radius (or radii) of curvature of the wavefront it emits, overriding
whatever :attr:`GeometricFieldSolver.tx_wavefront_radii` is set to: a
single value describes a spherical wavefront (like :class:`Dipole`), a
``(rho_s, rho_p)`` tuple an astigmatic one, and :data:`None` a planar one
(the far-field, plane-wave approximation). Subclass
:class:`AbstractFarFieldAntenna` (instead of :class:`AbstractAntenna` directly) for an
antenna that is only ever used in the far field, to get this last case
for free; :class:`FarFieldDipoleAntenna` does exactly that for
:class:`Dipole`.

.. autosummary::
   :toctree: _autosummary

   BaseAntenna
   AbstractAntenna
   Dipole
   AbstractFarFieldAntenna
   FarFieldDipoleAntenna

.. rubric:: Materials

We provide a basic class to represent radio materials,
and a mapping containing some common materials (e.g., ITU-R materials).

.. currentmodule:: differt.em._material

.. autosummary::
   :toctree: _autosummary

   Material
   MaterialsDict
   materials
   materials_from_scene
   AbstractScatteringPattern
   LambertianPattern
   DirectivePattern
   BackscatteringPattern

.. itu-materials-table::

.. currentmodule:: differt.em

Types of interaction (reflection, diffraction, etc.) within a path
are identified by different numbers, which are listed in an enum class.

.. autosummary::
   :toctree: _autosummary

   InteractionType

.. rubric:: Field solvers

Field solvers compute the received complex field(s) from a set of paths and
the geometry/materials they interacted with. :class:`GeometricFieldSolver`
is the default solver used by :func:`compute_received_fields`; subclass it
to customize how each interaction type contributes to the field.
Solver-specific configuration (antenna polarization, radio materials,
transmitter wavefront curvature) lives on the solver instance itself, as
plain attributes, rather than as keyword arguments to
:meth:`~GeometricFieldSolver.compute_fields`.

By default, :class:`GeometricFieldSolver` supports all four
:class:`InteractionType` members: reflection and transmission (a
finite-thickness dielectric slab model), diffraction (the Uniform Theory
of Diffraction), and diffuse scattering (a deterministic adaptation of a
Lambertian rough-surface model); see each ``*_matrix`` method's docstring
for details and caveats, particularly for scattering.

Unlike Sionna RT, which only supports a point source infinitely far away,
:attr:`GeometricFieldSolver.tx_wavefront_radii` supports a non-planar
(near-field) source, e.g., a focused beam, either isotropic (a single
radius, spherical wavefront) or astigmatic (a ``(rho_s, rho_p)`` tuple of
two independent principal radii); or, when :attr:`~GeometricFieldSolver.tx_polarization`
is set to an :class:`AbstractAntenna` instance, whatever that antenna's own
:meth:`AbstractAntenna.wavefront_radii` reports.

.. autosummary::
   :toctree: _autosummary

   AbstractFieldSolver
   GeometricFieldSolver

.. rubric:: Pipelines

End-to-end pipelines and high-level wrappers to compute received fields, Channel Impulse Response (CIR), and received power.

.. autosummary::
   :toctree: _autosummary

   TracedFields
   compute_cir
   compute_received_fields
   compute_received_power
   diffraction_matrix
   reflection_matrix
   ris_matrix
   scattering_matrix
   transition_matrix
   transmission_matrix

.. rubric:: Utilities

Utility functions, mostly used internally for computing EM fields.

.. autosummary::
   :toctree: _autosummary

   fspl
   length_to_delay
   path_delay
   poynting_vector
   sp_directions
   sp_rotation_matrix

.. rubric:: Work in progress

The following utilities are still under development, and using them is not recommended.

.. autosummary::
   :toctree: _autosummary

   ShortDipole
   AbstractRadiationPattern
   HWDipolePattern
   ShortDipolePattern

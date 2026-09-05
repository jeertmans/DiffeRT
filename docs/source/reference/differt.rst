``differt`` package
===================

.. currentmodule:: differt

.. automodule:: differt

The top-level ``differt`` package lazily re-exports (:pep:`562`) key classes,
functions, and interaction types from :mod:`differt.geometry` and
:mod:`differt.em`. Accessing any of these names imports the defining submodule on
first use, keeping ``import differt`` alone fast and lightweight.

.. rubric:: Re-exported Geometry Classes

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Top-Level Name
     - Canonical Location & Description
   * - ``differt.Scene``
     - :class:`~differt.geometry.Scene`: Scene containing meshes, transmitters, and receivers.
   * - ``differt.Mesh``
     - :class:`~differt.geometry.Mesh`: Triangle mesh geometry with radio materials.
   * - ``differt.TracedPaths``
     - :class:`~differt.geometry.TracedPaths`: Traced ray paths connecting transmitters to receivers.
   * - ``differt.LaunchedPaths``
     - :class:`~differt.geometry.LaunchedPaths`: Paths launched by shooting-and-bouncing-rays (SBR).

.. rubric:: Re-exported Electromagnetics and Wavefronts

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Top-Level Name
     - Canonical Location & Description
   * - ``differt.Material``
     - :class:`~differt.em._material.Material`: Radio material properties (permittivity, conductivity, roughness).
   * - ``differt.InteractionType``
     - :class:`~differt.em.InteractionType`: Enum flags indicating interaction mechanisms (reflection, diffraction, etc.).
   * - ``differt.WavefrontState``
     - :class:`~differt.em.WavefrontState`: Astigmatic wavefront curvature state (radii, axes, planarity).
   * - ``differt.propagate_wavefront``
     - :func:`~differt.em.propagate_wavefront`: Propagates wavefront curvature state along traced paths.

.. rubric:: Re-exported Interaction Type Aliases

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Top-Level Name
     - Canonical Location & Description
   * - ``differt.SpecularReflection``
     - :data:`~differt.em.SpecularReflection`: Alias for :attr:`InteractionType.REFLECTION <differt.em.InteractionType.REFLECTION>`.
   * - ``differt.Diffraction``
     - :data:`~differt.em.Diffraction`: Alias for :attr:`InteractionType.DIFFRACTION <differt.em.InteractionType.DIFFRACTION>`.
   * - ``differt.Scattering``
     - :data:`~differt.em.Scattering`: Alias for :attr:`InteractionType.SCATTERING <differt.em.InteractionType.SCATTERING>`.
   * - ``differt.Transmission``
     - :data:`~differt.em.Transmission`: Alias for :attr:`InteractionType.TRANSMISSION <differt.em.InteractionType.TRANSMISSION>`.
   * - ``differt.RIS``
     - :data:`~differt.em.RIS`: Alias for :attr:`InteractionType.RIS <differt.em.InteractionType.RIS>`.

Submodules
----------

.. toctree::
   :maxdepth: 1

   differt.em
   differt.geometry
   differt.plotting
   differt.plugins
   differt.utils

``differt.geometry`` module
===========================

.. currentmodule:: differt.geometry

.. automodule:: differt.geometry

.. rubric:: Scene and Meshes

The most important classes for defining the environment.

.. autosummary::
   :toctree: _autosummary

   Scene
   Mesh

.. rubric:: Reusing scenes from Sionna

Provide a compatibility layer with Sionna's scenes :cite:`sionna`.

Sionna uses the simple XML-based format from Mitsuba 3.

.. autosummary::
   :toctree: _autosummary

   download_sionna_scenes
   get_sionna_scene
   list_sionna_scenes

.. rubric:: Geometry Utilities

Utilities to transform 3D coordinates and miscellaneous helpers.

.. autosummary::
   :toctree: _autosummary

   assemble_path
   cartesian_to_spherical
   spherical_to_cartesian
   rotation_matrix_along_axis
   rotation_matrix_along_x_axis
   rotation_matrix_along_y_axis
   rotation_matrix_along_z_axis
   fibonacci_lattice
   merge_cell_ids
   min_distance_between_cells
   normalize
   perpendicular_vector
   orthogonal_basis
   path_length
   triangle_contains_vertex_assuming_inside_same_plane
   viewing_frustum

.. rubric:: Ray Tracing

Methods and classes for simulating electromagnetic wave propagation, path tracing, and ray launching.

.. autosummary::
   :toctree: _autosummary

   TracedPaths
   LaunchedPaths

.. rubric:: Path solvers

.. autosummary::
   :toctree: _autosummary

   AbstractPathSolver
   AbstractPathTracer
   AbstractPathLauncher
   ExhaustivePathTracer
   HybridPathTracer
   SBRPathLauncher

.. rubric:: Image method

Image-based path tracing.

.. autosummary::
   :toctree: _autosummary

   image_method
   image_of_vertex_with_respect_to_mirror
   intersection_of_ray_with_plane

.. rubric:: Fermat path tracing

Path tracing utilities that utilize Fermat's principle.

Fermat's principle states that the path taken by a ray between two
given points is the path that can be traveled in the least time
:cite:`fermat-principle`. In a homogeneous medium,
this means that the path of least time is also the path of least distance.

As a result, this module offers minimization methods for finding ray paths.

.. autosummary::
   :toctree: _autosummary

   fermat_path_on_linear_objects
   fermat_path_on_planar_mirrors

.. rubric:: Path candidates iterators

Useful utilities to generate path candidates, see :ref:`path_candidates`.

To generate a subset of all paths between two vertices, e.g.,
a transmitter TX and a receiver RX, path tracing methods generate
each ray path from a corresponding path candidate.

A path candidate is simply a list of primitive indices
to indicate with what primitive the path interacts, and
in what order. The latter indicates that any permutation
of a given path candidate will result in another path.

I.e., the path candidate ``[4, 7]`` indicates that
the path first interacts with primitive ``4``, then
primitive ``7``, while the path candidate ``[7, 4]``
indicates a path interacting first with ``7`` then
with ``4``.

An empty path candidate indicates a direct path from
TX or RX, also known as line-of-sight path.

In general, interaction can be anything of the following:
reflection, diffraction, refraction, etc. The utilities
present in this module do not take care of the interaction type.

For fine tuning, use :mod:`differt_core.geometry`'s graphs and iterators.

You can also read more about path candidates in :cite:`mpt-eucap2023`.

.. autosummary::
   :toctree: _autosummary

   generate_all_path_candidates
   generate_all_path_candidates_chunks_iter
   generate_all_path_candidates_iter

All returned iterators are of the following type.

.. autoclass:: SizedIterator

.. rubric:: Sanity checks

Utilities to check that ray paths are physically valid.

.. autosummary::
   :toctree: _autosummary

   consecutive_vertices_are_on_same_side_of_mirror
   first_triangle_hit_by_ray
   ray_intersect_any_triangle
   ray_intersect_triangle
   triangles_visible_from_vertex

.. rubric:: Deprecated classes

.. autosummary::
   :toctree: _autosummary

   Paths
   SBRPaths
   TriangleMesh
   TriangleScene

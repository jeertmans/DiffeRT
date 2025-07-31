``differt.rt`` module
=====================

.. currentmodule:: differt.rt

.. automodule:: differt.rt

.. rubric:: Image method

Image-based path tracing.

.. autosummary::
   :toctree: _autosummary

   image_method
   image_of_vertices_with_respect_to_mirrors
   intersection_of_rays_with_planes

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

For fine tuning, use :mod:`differt_core.rt`'s graphs and iterators.

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

   consecutive_vertices_are_on_same_side_of_mirrors
   first_triangles_hit_by_rays
   rays_intersect_any_triangle
   rays_intersect_triangles
   triangles_visible_from_vertices

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
this means that the path of least time is also the path of last distance.

As a result, this module offers minimization methods for finding ray paths.

.. autosummary::
   :toctree: _autosummary

   fermat_path_on_planar_mirrors

.. rubric:: Path candidates iterators

Useful utilities to generate path candidates, see :ref:`path_candidates`.

For fine tuning, use :mod:`differt_core.rt`'s graphs and iterators.

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
   rays_intersect_any_triangle
   rays_intersect_triangles
   triangles_visible_from_vertices

``differt.rt`` module
=====================

.. currentmodule:: differt.rt

.. automodule:: differt.rt

.. rubric:: Image method

.. automodule:: differt.rt._image_method

.. autosummary::
   :toctree: _autosummary

   image_method
   image_of_vertices_with_respect_to_mirrors
   intersection_of_rays_with_planes

.. rubric:: Fermat path tracing

.. automodule:: differt.rt._fermat

.. autosummary::
   :toctree: _autosummary

   fermat_path_on_planar_mirrors

.. currentmodule:: differt.rt

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

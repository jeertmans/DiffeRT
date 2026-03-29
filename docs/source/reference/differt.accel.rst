``differt.accel`` module
========================

.. currentmodule:: differt.accel

.. automodule:: differt.accel

.. rubric:: BVH acceleration structure

.. autosummary::
   :toctree: _autosummary

   TriangleBvh

.. rubric:: BVH-accelerated intersection functions

Drop-in replacements for :mod:`differt.rt` intersection functions
that use a BVH for O(log N) spatial queries instead of brute-force O(N).

.. autosummary::
   :toctree: _autosummary

   bvh_first_triangles_hit_by_rays
   bvh_rays_intersect_any_triangle
   bvh_triangles_visible_from_vertices

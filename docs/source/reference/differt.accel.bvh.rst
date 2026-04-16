``differt.accel.bvh`` module
============================

.. currentmodule:: differt.accel.bvh

.. automodule:: differt.accel.bvh

.. rubric:: BVH acceleration structure

BVH acceleration structure wrapping the Rust implementation.

Provides :class:`TriangleBvh` for accelerating ray-triangle intersection queries
from :math:`\mathcal{O}(\text{rays} \cdot \text{triangles})` to :math:`\mathcal{O}(\text{rays} \cdot \log(\text{triangles}))`.

.. autosummary::
   :toctree: _autosummary

   TriangleBvh

.. rubric:: BVH-accelerated intersection functions

Drop-in replacements for the functions in :mod:`differt.rt`,
accelerated by a BVH for :math:`\mathcal{O}(\text{rays} \cdot \log(\text{triangles}))` instead of :math:`\mathcal{O}(\text{rays} \cdot \text{triangles})`.

Without smoothing (``smoothing_factor=None``), the BVH does the full intersection.
With smoothing (``smoothing_factor`` set), the BVH selects candidates and the
existing JAX-based Möller-Trumbore runs on the reduced set.

.. autosummary::
   :toctree: _autosummary

   bvh_first_triangles_hit_by_rays
   bvh_rays_intersect_any_triangle
   bvh_triangles_visible_from_vertices

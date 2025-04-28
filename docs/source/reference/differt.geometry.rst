``differt.geometry`` module
===========================

.. currentmodule:: differt.geometry

.. automodule:: differt.geometry

.. rubric:: Classes

.. autosummary::
   :toctree: _autosummary

   Paths
   SBRPaths
   TriangleMesh


.. rubric:: Transformations

Utilities to transform 3D coordinates.

.. autosummary::
   :toctree: _autosummary

   rotation_matrix_along_axis
   rotation_matrix_along_x_axis
   rotation_matrix_along_y_axis
   rotation_matrix_along_z_axis
   cartesian_to_spherical
   spherical_to_cartesian

.. rubric:: Misc

Miscellaneous utilities.

.. autosummary::
   :toctree: _autosummary

   assemble_paths
   fibonacci_lattice
   merge_cell_ids
   min_distance_between_cells
   normalize
   pairwise_cross
   perpendicular_vectors
   orthogonal_basis
   path_lengths
   triangles_contain_vertices_assuming_inside_same_plane
   viewing_frustum

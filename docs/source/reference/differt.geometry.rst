``differt.geometry`` module
===========================

.. currentmodule:: differt.geometry

.. automodule:: differt.geometry

.. rubric:: Classes

.. autosummary::
   :toctree: _autosummary

   Paths
   SBRPaths
   TracePaths
   LaunchPaths
   TriangleMesh
   TriangleScene
   PathSolverConfig
   ExhaustivePathSolver
   HybridPathSolver
   SBRPathSolver

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

.. rubric:: Image method

Image-based path tracing.

.. autosummary::
   :toctree: _autosummary

   image_method
   image_of_vertex_with_respect_to_mirror
   intersection_of_ray_with_plane

.. rubric:: Fermat path tracing

Path tracing utilities that utilize Fermat's principle.

.. autosummary::
   :toctree: _autosummary

   fermat_path_on_linear_objects
   fermat_path_on_planar_mirrors

.. rubric:: Path candidates iterators

Useful utilities to generate path candidates.

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

.. rubric:: Reusing scenes from Sionna

Provide a compatibility layer with Sionna's scenes.

.. autosummary::
   :toctree: _autosummary

   download_sionna_scenes
   get_sionna_scene
   list_sionna_scenes

.. rubric:: Misc

Miscellaneous utilities.

.. autosummary::
   :toctree: _autosummary

   assemble_path
   fibonacci_lattice
   merge_cell_ids
   min_distance_between_cells
   normalize
   perpendicular_vector
   orthogonal_basis
   path_length
   triangle_contains_vertex_assuming_inside_same_plane
   viewing_frustum

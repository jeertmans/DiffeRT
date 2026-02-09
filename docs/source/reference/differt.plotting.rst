``differt.plotting`` module
===========================

.. currentmodule:: differt.plotting

.. automodule:: differt.plotting

.. rubric:: Common utils

The following utilities allow you to modify the default behavior
of plotting functions, either permanently or inside a given scope.

.. autosummary::
   :toctree: _autosummary

   reuse
   set_backend
   set_defaults
   update_defaults
   use

.. rubric:: Drawing functions

List of all drawing functions provided by this module.

.. note::
   Some functions might not support all three backends. In such cases, it is indicated in their documentation.

.. autosummary::
   :toctree: _autosummary

   draw_contour
   draw_image
   draw_markers
   draw_mesh
   draw_paths
   draw_rays
   draw_surface

.. rubric:: Extending this module

If you want to add new drawing functions, either locally or by contributing to this package,
the following utilities might become handy.

.. autosummary::
   :toctree: _autosummary

   dispatch
   get_backend
   process_kwargs
   process_matplotlib_kwargs
   process_plotly_kwargs
   process_vispy_kwargs
   view_from_canvas

Note that all higher-level plotting functions, i.e., created outside of this module,
should have the following return type.

.. autoclass:: PlotOutput

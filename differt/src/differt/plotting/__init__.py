"""
Plotting utilities for DiffeRT objects.

.. tip::

    Unlike in other modules, plotting utilities also work
    with NumPy arrays (:class:`np.ndarray<numpy.ndarray>`)


.. note::

    Backend names are **case-sensitive**.


Backend-specific notes
======================

Each backend comes with its own configuration options and outputs
that the user should be aware of.

VisPy
-----

VisPy uses :class:`SceneCanvas<vispy.scene.canvas.SceneCanvas>` objects
to display contents, on which a view is attached. The view
(:class:`ViewBox<vispy.scene.widgets.viewbox.ViewBox>`)
is what contains
the data to be plotted.

To reuse an existing ``canvas`` object, just pass it as a keyword
argument to any of the ``draw_*`` functions in this module, i.e.,
with ``draw_*(..., canvas=canvas)``.
In turn, each of those functions returns a figure on which you
can later add data.

It is also possible to pass an existing view
on which data will be plotted: ``draw_*(..., view=view)``.

If the ``jupyter_rfb`` module is installed, VisPy's canvas integrate
nicely within Jupyter notebooks.

Matplotlib
----------

Matplotlib uses :class:`Figure<matplotlib.figure.Figure>` objects
to display contents, on which multiple axes can be attached. In turn, each
axis can contain data to be plotted.

To reuse an existing ``figure`` object, just pass it as a keyword
argument to any of the ``draw_*`` functions in this module, i.e.,
with ``draw_*(..., figure=figure)``.
In turn, each of those functions returns a figure on which you
can later add data.

It is also possible to pass an existing axis
(:class:`Axes<matplotlib.axes.Axes>`)
on which data will be plotted: ``draw_*(..., ax=ax)``.

.. warning::

    By default, Matplotlib instantiates 2D axes, but this module
    extensively uses 3D plot methods. If an axis is provided,
    it is your responsibility to ensure that it can plot 3D data
    when needed.

By default, Matpotlib displays static images in Jupyter notebooks.
To make them interactive, install ``ipympl`` and load the corresponding
extension with ``%matplotlib widget``.

Plotly
------

Plotly is a dictionary-oriented library that produces HTML-based
outputs. Hence, Plotly is a very good choice for publishing
nice interactive plots on webpages.

Plots are fully contained inside
:class:`Figure<plotly.graph_objects.Figure>` objects, and can be nicely
displayed within Jupyter notebooks without further configuration.

To reuse an existing ``figure`` object, you can do the same as with
the Matplotlib backend.

"""

__all__ = (
    "PlotOutput",
    "dispatch",
    "draw_contour",
    "draw_image",
    "draw_markers",
    "draw_mesh",
    "draw_paths",
    "draw_rays",
    "draw_surface",
    "get_backend",
    "process_matplotlib_kwargs",
    "process_plotly_kwargs",
    "process_vispy_kwargs",
    "reuse",
    "set_defaults",
    "use",
    "view_from_canvas",
)

from ._core import (
    PlotOutput,
    draw_contour,
    draw_image,
    draw_markers,
    draw_mesh,
    draw_paths,
    draw_rays,
    draw_surface,
)
from ._utils import (
    dispatch,
    get_backend,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
    reuse,
    set_defaults,
    use,
    view_from_canvas,
)

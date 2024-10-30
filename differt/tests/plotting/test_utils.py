from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure as MplFigure
from plotly.graph_objs import Figure
from vispy.scene.canvas import SceneCanvas

from differt.plotting import (
    dispatch,
    get_backend,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
    reuse,
    use,
    view_from_canvas,
)

if TYPE_CHECKING:
    from pytest_missing_modules.plugin import MissingModulesContextGenerator


@dispatch  # type: ignore[reportArgumentType]
def my_plot_unimplemented(**kwargs: Any) -> SceneCanvas | MplFigure | Figure:  # type: ignore[reportReturnType]
    """A plot function with no backend implementation."""


@dispatch  # type: ignore[reportArgumentType]
def my_plot(**kwargs: Any) -> SceneCanvas | MplFigure | Figure:  # type: ignore[reportReturnType]
    """A plot function with dummy backend implementations."""


@my_plot.register("vispy")
def _(**kwargs: Any) -> SceneCanvas:
    canvas, _view = process_vispy_kwargs(kwargs)
    return canvas


@my_plot.register("matplotlib")
def _(**kwargs: Any) -> MplFigure:
    fig, _ax = process_matplotlib_kwargs(kwargs)
    return fig


@my_plot.register("plotly")
def _(**kwargs: Any) -> Figure:
    return process_plotly_kwargs(kwargs)


@pytest.mark.parametrize("backend", [None, "vispy", "matplotlib", "plotly"])
def test_unimplemented(backend: str | None) -> None:
    if backend:
        with pytest.raises(
            NotImplementedError, match=f"No backend implementation for '{backend}'"
        ):
            _ = my_plot_unimplemented(backend=backend)
    else:
        with pytest.raises(NotImplementedError, match="No backend implementation for"):
            _ = my_plot_unimplemented()


def test_use_unsupported() -> None:
    with (
        pytest.raises(
            ValueError,
            match="The backend 'bokeh' is not supported. We currently support:",
        ),
        use(backend="bokeh"),
    ):
        pass


def test_register_unsupported() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported backend 'bokeh', allowed values are:",
    ):

        @my_plot.register("bokeh")
        def _(**_kwargs: Any) -> None:
            pass


@pytest.mark.parametrize("backend", ["vispy", "matplotlib", "plotly"])
def test_missing_default_backend_module(
    backend: str,
    missing_modules: MissingModulesContextGenerator,
) -> None:
    with use(backend=backend):  # Change the default backend
        with (
            missing_modules(backend),
            pytest.raises(
                ImportError,
                match="An import error occurred when dispatching plot utility "
                f"to backend '{backend}'. Did you correctly install it?",
            ),
        ):
            _ = my_plot()

    with (
        missing_modules(backend),
        pytest.raises(
            ImportError,
            match=f"Could not find backend '{backend}', did you install it?",
        ),
    ):
        with use(backend=backend):
            pass


@pytest.mark.parametrize(
    "backend",
    ["vispy", "matplotlib", "plotly"],
)
def test_missing_backend_module(
    backend: str,
    missing_modules: MissingModulesContextGenerator,
) -> None:
    with (
        missing_modules(backend),
        pytest.raises(
            ImportError,
            match=f"Could not find backend '{backend}', did you install it?",
        ),
    ):
        _ = my_plot(backend=backend)


@pytest.mark.parametrize(
    ("backend", "rtype"),
    [("vispy", SceneCanvas), ("matplotlib", MplFigure), ("plotly", Figure)],
)
def test_return_type(backend: str, rtype: type) -> None:
    ret = my_plot(backend=backend)
    assert isinstance(ret, rtype), f"{ret!r} is not of type {rtype}"


def test_process_vispy_kwargs() -> None:
    kwargs: dict[str, Any] = {"color": "red"}
    canvas, view = process_vispy_kwargs(kwargs)
    assert view == view_from_canvas(canvas)

    kwargs["canvas"] = canvas

    got_canvas, got_view = process_vispy_kwargs(kwargs)
    assert canvas == got_canvas
    assert view == got_view
    assert "canvas" not in kwargs

    kwargs["view"] = view

    got_canvas, got_view = process_vispy_kwargs(kwargs)
    assert canvas == got_canvas
    assert view == got_view
    assert "view" not in kwargs

    kwargs["canvas"] = canvas
    kwargs["view"] = view

    got_canvas, got_view = process_vispy_kwargs(kwargs)
    assert canvas == got_canvas
    assert view == got_view
    assert "canvas" not in kwargs
    assert "view" not in kwargs


def test_process_matplotlib_kwargs() -> None:
    kwargs: dict[str, Any] = {"color": "green"}
    fig, ax = process_matplotlib_kwargs(kwargs)

    kwargs["figure"] = fig

    plt.figure()  # Let's create another figure to pollute

    got_fig, got_ax = process_matplotlib_kwargs(kwargs)
    assert fig == got_fig
    assert ax == got_ax
    assert "figure" not in kwargs

    kwargs["ax"] = ax

    got_fig, got_ax = process_matplotlib_kwargs(kwargs)
    assert fig == got_fig
    assert ax == got_ax
    assert "ax" not in kwargs

    kwargs["figure"] = fig
    kwargs["ax"] = ax

    got_fig, got_ax = process_matplotlib_kwargs(kwargs)
    assert fig == got_fig
    assert ax == got_ax
    assert "figure" not in kwargs
    assert "ax" not in kwargs


def test_process_plotly_kwargs() -> None:
    kwargs: dict[str, Any] = {"color": "blue"}
    fig = process_plotly_kwargs(kwargs)

    kwargs["figure"] = fig

    got_fig = process_plotly_kwargs(kwargs)
    assert fig == got_fig
    assert "figure" not in kwargs


@pytest.mark.parametrize("backend", [None, "vispy", "matplotlib", "plotly"])
def test_reuse(backend: str) -> None:
    expected = {"vispy": SceneCanvas, "matplotlib": MplFigure, "plotly": Figure}[
        get_backend(backend)
    ]

    with reuse(backend) as got:
        assert isinstance(got, expected)

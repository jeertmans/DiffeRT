from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import AbstractContextManager as ContextManager
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Protocol

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure as MplFigure
from plotly.graph_objs import Figure
from vispy.scene.canvas import SceneCanvas

from differt.plotting._utils import (
    dispatch,
    process_matplotlib_kwargs,
    process_plotly_kwargs,
    process_vispy_kwargs,
    use,
    view_from_canvas,
)


class MissingModulesContextGenerator(Protocol):
    def __call__(self, *names: str) -> ContextManager[pytest.MonkeyPatch]:
        ...


@pytest.fixture
def missing_modules(monkeypatch: pytest.MonkeyPatch) -> MissingModulesContextGenerator:
    @contextmanager  # type: ignore[arg-type]
    def ctx(*names: str) -> ContextManager[pytest.MonkeyPatch]:  # type: ignore[misc]
        real_import = builtins.__import__
        real_import_module = importlib.import_module

        def monkey_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
            if name in names:
                raise ImportError(f"Mocked import error for '{name}'")
            return real_import(name, *args, **kwargs)

        def monkey_import_module(name: str, *args: Any, **kwargs: Any) -> ModuleType:
            if name in names:
                raise ImportError(f"Mocked import error for '{name}'")
            return real_import_module(name, *args, **kwargs)

        with monkeypatch.context() as m:
            for name in names:
                m.delitem(sys.modules, name, raising=False)

            m.setattr(builtins, "__import__", monkey_import)
            m.setattr(importlib, "import_module", monkey_import_module)

            yield m  # type: ignore

    return ctx


@dispatch  # type: ignore
def my_plot_unimplemented(**kwargs: dict[str, Any]) -> SceneCanvas | MplFigure | Figure:  # type: ignore
    """A plot function with no backend implementation."""


@dispatch  # type: ignore
def my_plot(**kwargs: dict[str, Any]) -> SceneCanvas | MplFigure | Figure:  # type: ignore
    """A plot function with dummy backend implementations."""


@my_plot.register("vispy")
def _(**kwargs):  # type: ignore[no-untyped-def]
    canvas, view = process_vispy_kwargs(kwargs)
    return canvas


@my_plot.register("matplotlib")
def _(**kwargs):  # type: ignore[no-untyped-def]
    fig, ax = process_matplotlib_kwargs(kwargs)
    return fig


@my_plot.register("plotly")
def _(**kwargs):  # type: ignore[no-untyped-def]
    fig = process_plotly_kwargs(kwargs)
    return fig


@pytest.mark.parametrize("backend", (None, "vispy", "matplotlib", "plotly"))
def test_unimplemented(backend: str | None) -> None:
    with pytest.raises(NotImplementedError, match="No backend implementation for"):
        if backend:
            _ = my_plot_unimplemented(backend=backend)  # type: ignore
        else:
            _ = my_plot_unimplemented()


def test_use_unsupported() -> None:
    with pytest.raises(
        ValueError, match="The backend 'bokeh' is not supported. We currently support:"
    ):
        use("bokeh")


def test_register_unsupported() -> None:
    with pytest.raises(
        ValueError, match="Unsupported backend 'bokeh', allowed values are:"
    ):

        @my_plot.register("bokeh")
        def _(**kwargs):  # type: ignore[no-untyped-def]
            pass


@pytest.mark.parametrize("backend", ("vispy", "matplotlib", "plotly"))
def test_missing_default_backend_module(
    backend: str, missing_modules: MissingModulesContextGenerator
) -> None:
    use(backend)  # Change the default backend

    with missing_modules(backend), pytest.raises(
        ImportError,
        match=f"An import error occured when dispatching plot utility to backend '{backend}'.",
    ):
        _ = my_plot()

    with missing_modules(backend), pytest.raises(
        ImportError, match=f"Could not load backend '{backend}'"
    ):
        use(backend)


@pytest.mark.parametrize("backend", ("vispy", "matplotlib", "plotly"))
def test_missing_backend_module(
    backend: str, missing_modules: MissingModulesContextGenerator
) -> None:
    with missing_modules(backend), pytest.raises(
        ImportError,
        match=f"An import error occured when dispatching plot utility to backend '{backend}'.",
    ):
        _ = my_plot(backend=backend)  # type: ignore


@pytest.mark.parametrize(
    "backend,rtype",
    (("vispy", SceneCanvas), ("matplotlib", MplFigure), ("plotly", Figure)),
)
def test_return_type(backend: str, rtype: type) -> None:
    ret = my_plot(backend=backend)  # type: ignore
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

    got_figure, got_ax = process_matplotlib_kwargs(kwargs)
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

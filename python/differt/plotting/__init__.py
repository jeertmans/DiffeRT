"""
Plotting utils for DiffeRT objects.
"""
from __future__ import annotations

import importlib

import numpy as np
from jaxtyping import Float, UInt
from typing import TYPE_CHECKING

from .decorators import dispatch

if TYPE_CHECKING:
    from vispy.app.canvas import Canvas
    from plotly.graph_objects import Figure


CURRENT_BACKEND = None
DEFAULT_BACKEND = "vispy"
SUPPORTED_BACKENDS = ("vispy", "matplotlib", "plotly")


class Plotter:
    def __init__(self, *args, **kwargs):
        self.backend = load(kwargs.pop("backend", None))

    def mesh(self, *args, **kwargs) -> Canvas:
        from .vispy import visuals

        mesh = visuals.Mesh(*args, **kwargs)


def mesh(vertices: np.ndarray | None = None, faces: np.ndarray | None = None) -> Canvas | Figure:
    """
    Plot a 3D mesh made of triangles.
    """
    pass
        

def use(backend: str) -> None:
    """
    Tell future plotting utilities to use this backend by default.

    Args:
        backend: The name of the backend to use.

    Raises:
        ValueError: If the backend is not supported.
        ImportError: If the backend is not installed.
    """
    global DEFAULT_BACKEND

    load(backend)
        
    DEFAULT_BACKEND = backend
   

def load(backend: str | None = None) -> str:
    """
    Load the given backend module, or the default one.

    Args:
        backend: The name of the backend to use.

    Returns:
        The backend that was loaded.

    Raises:
        ValueError: If the backend is not supported.
        ImportError: If the backend is not installed.
        ModuleNotFound: TODO.
    """

    if backend or DEFAULT_BACKEND:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"The backend '{backend}' is not supported. "
                f"We currently support '{SUPPORTED_BACKENDS}'")

        try:
            importlib.import_module(f"{backend}")
            return backend
        except ImportError:
            raise ImportError(f"Could not load backend '{backend}', did you install it?") from None

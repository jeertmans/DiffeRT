"""Differentiable Ray Tracing Toolbox for Radio Propagation."""

import importlib
from typing import TYPE_CHECKING

from ._version import __version__, __version_info__

if TYPE_CHECKING:
    from .em import (
        RIS,
        Diffraction,
        InteractionType,
        Material,
        Scattering,
        SpecularReflection,
        Transmission,
        WavefrontState,
        propagate_wavefront,
    )
    from .geometry import LaunchedPaths, Mesh, Scene, TracedPaths

__all__ = (
    "RIS",
    "Diffraction",
    "InteractionType",
    "LaunchedPaths",
    "Material",
    "Mesh",
    "Scattering",
    "Scene",
    "SpecularReflection",
    "TracedPaths",
    "Transmission",
    "WavefrontState",
    "__version__",
    "__version_info__",
    "propagate_wavefront",
)

# Lazily re-export the most commonly used names from 'differt.geometry' and
# 'differt.em' at the top level (PEP 562), so 'import differt' stays cheap:
# neither submodule (and none of their heavy dependencies, e.g. 'warp' or
# 'fpt_jax') is imported until one of these names is actually accessed.
# This also sidesteps the geometry <-> em circular import (see
# 'Scene.load_xml'/'Scene.trace_fields' for its existing occurrences).
_LAZY = {
    "Scene": ".geometry",
    "Mesh": ".geometry",
    "TracedPaths": ".geometry",
    "LaunchedPaths": ".geometry",
    "Material": ".em",
    "InteractionType": ".em",
    "SpecularReflection": ".em",
    "Diffraction": ".em",
    "Scattering": ".em",
    "Transmission": ".em",
    "RIS": ".em",
    "WavefrontState": ".em",
    "propagate_wavefront": ".em",
}


def __getattr__(name: str) -> object:
    if mod := _LAZY.get(name):
        return getattr(importlib.import_module(mod, __name__), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return [*globals(), *_LAZY]

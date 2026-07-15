"""Deprecated module."""

# ruff:file-ignore[non-empty-init-module, module-import-not-at-top-of-file]
import warnings

warnings.warn(
    "The differt.scene module is deprecated and will be removed in a future version. "
    "Please use differt.geometry instead.",
    DeprecationWarning,
    stacklevel=2,
)

from differt.geometry import (
    AbstractPathLauncher,
    AbstractPathSolver,
    AbstractPathTracer,
    ExhaustivePathTracer,
    HybridPathTracer,
    Material,
    SBRPathLauncher,
    Scene,
    Shape,
    SionnaScene,
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
    list_sionna_scenes,
)

__all__ = (
    "AbstractPathLauncher",
    "AbstractPathSolver",
    "AbstractPathTracer",
    "ExhaustivePathTracer",
    "HybridPathTracer",
    "Material",
    "SBRPathLauncher",
    "Scene",
    "Shape",
    "SionnaScene",
    "TriangleScene",
    "download_sionna_scenes",
    "get_sionna_scene",
    "list_sionna_scenes",
)

"""Geometry utilities used by :mod:`differt.geometry`."""

from differt_core import _differt_core

__all__ = (
    "AllPathsFromCompleteGraphChunksIter",
    "AllPathsFromCompleteGraphIter",
    "AllPathsFromDiGraphChunksIter",
    "AllPathsFromDiGraphIter",
    "CompleteGraph",
    "DiGraph",
    "Material",
    "Shape",
    "SionnaScene",
    "TriangleMesh",
    "TriangleScene",
)

TriangleMesh = _differt_core.geometry.triangle_mesh.TriangleMesh

AllPathsFromCompleteGraphChunksIter = (
    _differt_core.rt.graph.AllPathsFromCompleteGraphChunksIter
)
AllPathsFromCompleteGraphIter = _differt_core.rt.graph.AllPathsFromCompleteGraphIter
AllPathsFromDiGraphChunksIter = _differt_core.rt.graph.AllPathsFromDiGraphChunksIter
AllPathsFromDiGraphIter = _differt_core.rt.graph.AllPathsFromDiGraphIter
CompleteGraph = _differt_core.rt.graph.CompleteGraph
DiGraph = _differt_core.rt.graph.DiGraph

Material = _differt_core.scene.sionna.Material
Shape = _differt_core.scene.sionna.Shape
SionnaScene = _differt_core.scene.sionna.SionnaScene
TriangleScene = _differt_core.scene.triangle_scene.TriangleScene


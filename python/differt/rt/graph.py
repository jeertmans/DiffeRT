"""
Graph utitilies, re-exported from the Rust code module.

The present module is re-exported from the (private) core
module, for advanced usage only.

A highler-level interface is available in the :py:mod:`differt.rt.utils`
module.
"""

__all__ = (
    "CompleteGraph",
    "DiGraph",
    "AllPathsFromCompleteGraphIter",
    "AllPathsFromCompleteGraphChunksIter",
    "AllPathsFromDiGraphIter",
    "AllPathsFromDiGraphChunksIter",
)

from .. import _core

CompleteGraph = _core.rt.graph.CompleteGraph
DiGraph = _core.rt.graph.DiGraph
AllPathsFromCompleteGraphIter = _core.rt.graph.AllPathsFromCompleteGraphIter
AllPathsFromCompleteGraphChunksIter = _core.rt.graph.AllPathsFromCompleteGraphChunksIter
AllPathsFromDiGraphIter = _core.rt.graph.AllPathsFromDiGraphIter
AllPathsFromDiGraphChunksIter = _core.rt.graph.AllPathsFromDiGraphChunksIter

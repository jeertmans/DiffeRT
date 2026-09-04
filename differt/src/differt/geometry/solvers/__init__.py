"""Path solvers and launchers, split by strategy.

This subpackage is an internal implementation detail: import from
:mod:`differt.geometry` instead.
"""

from ._base import (
    AbstractPathLauncher,
    AbstractPathSolver,
    AbstractPathTracer,
    _generate_path_candidates_for_orders,
    _normalize_order,
    _pad_path_candidates,
    _trace_path_candidates,
)
from ._exhaustive import ExhaustivePathTracer, _ExhaustivePathTracerKwargs
from ._hybrid import HybridPathTracer, _HybridPathTracerKwargs
from ._sbr import (
    SBRPathLauncher,
    SBRPathTracer,
    _SBRPathLauncherKwargs,
    _SBRPathTracerKwargs,
)

__all__ = [
    "AbstractPathLauncher",
    "AbstractPathSolver",
    "AbstractPathTracer",
    "ExhaustivePathTracer",
    "HybridPathTracer",
    "SBRPathLauncher",
    "SBRPathTracer",
    "_ExhaustivePathTracerKwargs",
    "_HybridPathTracerKwargs",
    "_SBRPathLauncherKwargs",
    "_SBRPathTracerKwargs",
    "_generate_path_candidates_for_orders",
    "_normalize_order",
    "_pad_path_candidates",
    "_trace_path_candidates",
]

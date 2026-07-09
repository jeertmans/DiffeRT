import equinox as eqx
from jaxtyping import ArrayLike, Float


class PathSolverConfig(eqx.Module):
    """Base class for all path solver configurations."""


class ExhaustivePathSolver(PathSolverConfig):
    """Exhaustive (image-method) path solver.

    All possible path candidates are generated and tested.
    """

    epsilon: Float[ArrayLike, ""] | None = None
    hit_tol: Float[ArrayLike, ""] | None = None
    min_len: Float[ArrayLike, ""] | None = None
    smoothing_factor: Float[ArrayLike, ""] | None = None
    confidence_threshold: Float[ArrayLike, ""] = 0.5
    batch_size: int | None = 512
    disconnect_inactive_triangles: bool = False
    chunk_size: int | None = None


class HybridPathSolver(PathSolverConfig):
    """Hybrid path solver — SBR visibility + exhaustive tracing.

    Uses ray launching to estimate object visibility, then performs
    exhaustive search on the reduced candidate set.
    """

    num_rays: int = int(1e6)
    epsilon: Float[ArrayLike, ""] | None = None
    hit_tol: Float[ArrayLike, ""] | None = None
    min_len: Float[ArrayLike, ""] | None = None
    smoothing_factor: Float[ArrayLike, ""] | None = None
    confidence_threshold: Float[ArrayLike, ""] = 0.5
    batch_size: int | None = 512
    chunk_size: int | None = None


class SBRPathSolver(PathSolverConfig):
    """Shooting-and-bouncing ray (SBR) path solver."""

    num_rays: int = int(1e6)
    max_dist: Float[ArrayLike, ""] = 1e-3

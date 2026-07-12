import equinox as eqx
from jaxtyping import ArrayLike, Float


class PathSolverConfig(eqx.Module):
    """Base class for all path solver configurations."""


class ExhaustivePathSolver(PathSolverConfig):
    """
    Exhaustive (image-method) path solver.

    All possible path candidates are generated and tested. This is the slowest
    method, but it is also the most accurate.
    """

    epsilon: Float[ArrayLike, ""] | None = None
    """Tolerance for checking ray / object intersections."""
    hit_tol: Float[ArrayLike, ""] | None = None
    """Tolerance for blockage checks."""
    min_len: Float[ArrayLike, ""] | None = None
    """Minimal (squared) length that each path segment must have for a path to be valid."""
    smoothing_factor: Float[ArrayLike, ""] | None = None
    """Parameters for slope of the smoothing function."""
    confidence_threshold: Float[ArrayLike, ""] = 0.5
    """Confidence threshold for valid paths."""
    batch_size: int | None = 512
    """Intersection check batch size."""
    disconnect_inactive_triangles: bool = False
    """Whether to filter out inactive triangles first."""
    chunk_size: int | None = None
    """If specified, iterates through chunks of path candidates, yielding an iterator over path chunks."""
    max_candidates: int | None = None
    """Maximum number of path candidates to evaluate.

    If the graph produces more candidates than this limit, the array
    is truncated. This prevents out-of-memory errors on large scenes
    where the exhaustive :math:`O(N^d)` combinatorics become intractable.
    When combined with ``disconnect_inactive_triangles`` or the hybrid solver's
    visibility pruning, this provides a hard upper bound on memory usage.
    """


class HybridPathSolver(PathSolverConfig):
    """
    Hybrid path solver, combining ray launching for visibility and exhaustive tracing.

    Uses ray launching to estimate object visibility, then performs
    exhaustive search on the reduced candidate set. This is a faster
    alternative to exhaustive search, but still grows exponentially with
    the number of bounces or the size of the scene.

    .. warning::

        This method is best used for a single transmitter and a single receiver,
        as the estimated visibility is merged across all transmitters and receivers,
        respectively.
    """

    num_rays: int = int(1e6)
    """The number of rays launched."""
    epsilon: Float[ArrayLike, ""] | None = None
    """Tolerance for checking ray / object intersections."""
    hit_tol: Float[ArrayLike, ""] | None = None
    """Tolerance for blockage checks."""
    min_len: Float[ArrayLike, ""] | None = None
    """Minimal (squared) length that each path segment must have for a path to be valid."""
    smoothing_factor: Float[ArrayLike, ""] | None = None
    """Parameters for slope of the smoothing function."""
    confidence_threshold: Float[ArrayLike, ""] = 0.5
    """Confidence threshold for valid paths."""
    batch_size: int | None = 512
    """Intersection check batch size."""
    chunk_size: int | None = None
    """If specified, iterates through chunks of path candidates, yielding an iterator over path chunks."""


class SBRPathSolver(PathSolverConfig):
    """
    Shooting-and-bouncing ray (SBR) path solver.

    A fixed number of rays are launched from each transmitter and are allowed
    to perform a fixed number of bounces. Only ray paths passing in the vicinity
    of a receiver are considered valid.

    .. important::

        This SBR method is currently unstable and not yet optimized, and it is likely
        to change in future releases. Use with caution.
    """

    num_rays: int = int(1e6)
    """The number of rays launched."""
    max_dist: Float[ArrayLike, ""] = 1e-3
    """Maximal (squared) distance between a receiver and a ray for the receiver to be considered in the vicinity of the ray path."""

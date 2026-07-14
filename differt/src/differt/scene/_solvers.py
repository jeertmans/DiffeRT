import abc
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.geometry import (
    LaunchedPaths,
    TracedPaths,
    TriangleMesh,
    assemble_path,
    fibonacci_lattice,
    viewing_frustum,
)
from differt.rt import (
    SizedIterator,
    consecutive_vertices_are_on_same_side_of_mirror,
    image_method,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
)
from differt.utils import smoothing_function
from differt_core.rt import CompleteGraph, DiGraph

if TYPE_CHECKING:
    from differt.scene import TriangleScene


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class AbstractPathSolver(eqx.Module):
    """Base class for all path solvers and launchers.

    Subclasses should define concrete values for
    ``epsilon`` and ``hit_tol``.
    """

    # TODO: define default initialization for these in subclasses
    epsilon: AbstractVar[float]
    """Tolerance for checking ray / object intersections."""
    hit_tol: AbstractVar[float]
    """Tolerance for blockage checks."""


class AbstractPathTracer(AbstractPathSolver):
    """Abstract base class for exact path tracing solvers.

    A path tracer generates *path candidates* (arrays of triangle indices
    together with interaction-type tags) and then traces them through the
    scene to produce :class:`~differt.geometry.TracedPaths`.
    """

    @abc.abstractmethod
    def generate_path_candidates(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        specular_reflection: bool = True,
        diffuse_scattering: bool = False,
    ) -> tuple[
        Int[Array, "... num_path_candidates max_order"],
        Int[Array, "... num_path_candidates max_order"],
    ]:
        """
        Return a tuple of ``(path_candidates, interaction_types)``.

        ``path_candidates`` contains triangle indices.
        ``interaction_types`` classifies the bounce (e.g., ``0`` for specular).
        A value of ``-1`` in either array indicates an "inactive" interaction
        or padded bounce (used when combining paths of different reflection
        orders).

        Args:
            scene: The scene.
            order: The path order (number of bounces), or a sequence of
                orders to combine.
            specular_reflection: Whether to include specular reflections.
            diffuse_scattering: Whether to include diffuse scattering
                (not yet implemented).

        Returns:
            A 2-tuple of ``(path_candidates, interaction_types)``.
        """

    def generate_path_candidates_chunks_iter(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        *args: Any,
        chunk_size: int,
        pad_chunks: bool = False,
        **kwargs: Any,
    ) -> SizedIterator[
        tuple[
            Int[Array, "... chunk_size max_order"],
            Int[Array, "... chunk_size max_order"],
        ]
    ]:
        """Return an iterator of chunked path candidate tuples.

        The default implementation calls :meth:`generate_path_candidates`
        once and then slices the result. Subclasses may override this to
        generate candidates lazily.

        Args:
            scene: The scene.
            order: The path order(s).
            *args: Forwarded to :meth:`generate_path_candidates`.
            chunk_size: Number of candidates per chunk.
            pad_chunks: If ``True``, the last chunk is zero-padded
                (with ``-1``) to ``chunk_size``.
            **kwargs: Forwarded to :meth:`generate_path_candidates`.

        Returns:
            A :class:`~differt.rt.SizedIterator` over
            ``(path_candidates, interaction_types)`` chunks.
        """
        # Always expect a 2-tuple to keep JAX tree structures predictable
        candidates, interactions = self.generate_path_candidates(
            scene, order, *args, **kwargs
        )

        # Batch dimension is -2. Order dimension is -1.
        num_path_candidates = candidates.shape[-2]
        num_chunks, rem = divmod(num_path_candidates, chunk_size)
        total_chunks = num_chunks + (1 if rem > 0 else 0)

        def iter_chunks() -> Iterator[
            tuple[
                Int[Array, "... chunk_size max_order"],
                Int[Array, "... chunk_size max_order"],
            ]
        ]:
            # Yield full chunks
            for i in range(num_chunks):
                start_idx = i * chunk_size
                yield jax.tree.map(
                    lambda x, start_idx=start_idx: jax.lax.dynamic_slice_in_dim(
                        x, start_idx, chunk_size, axis=-2
                    ),
                    (candidates, interactions),
                )

            # Handle the remainder chunk
            if rem > 0:
                start_idx = num_chunks * chunk_size
                remainder_slice = jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, start_idx, rem, axis=-2),
                    (candidates, interactions),
                )

                if pad_chunks:

                    def pad_array(x: Array) -> Array:
                        # Pad only the num_path_candidates axis (-2)
                        pad_width = [(0, 0)] * x.ndim
                        pad_width[-2] = (0, chunk_size - rem)
                        return jnp.pad(
                            x, pad_width, mode="constant", constant_values=-1
                        )

                    yield jax.tree.map(pad_array, remainder_slice)
                else:
                    yield remainder_slice

        return SizedIterator(iter_chunks(), size=total_chunks)

    @abc.abstractmethod
    def trace_path_candidates(
        self,
        scene: "TriangleScene",
        path_candidates: Int[Array, "... num_path_candidates max_order"],
        interaction_types: Int[Array, "... num_path_candidates max_order"],
    ) -> TracedPaths:
        """Core logic to trace the exact paths from the proposed candidates.

        Args:
            scene: The scene.
            path_candidates: Triangle indices for each candidate.
            interaction_types: Interaction type for each bounce.

        Returns:
            The traced paths.
        """

    # -- Overloads for strict type hinting --

    @overload
    def trace_paths(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        chunk_size: None = None,
        pad_chunks: bool = False,
    ) -> TracedPaths: ...

    @overload
    def trace_paths(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        chunk_size: int,
        pad_chunks: bool = False,
    ) -> Iterator[TracedPaths]: ...

    def trace_paths(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        chunk_size: int | None = None,
        pad_chunks: bool = False,
    ) -> TracedPaths | Iterator[TracedPaths]:
        """
        Trace paths for the given scene and order(s).

        If ``chunk_size`` is provided, returns an iterator of
        :class:`~differt.geometry.TracedPaths` (one per chunk);
        otherwise returns a single :class:`~differt.geometry.TracedPaths`.

        Args:
            scene: The scene.
            order: The path order(s).
            chunk_size: If not ``None``, iterate through chunks of
                this size.
            pad_chunks: If ``True`` and ``chunk_size`` is set,
                pad the last chunk.

        Returns:
            Traced paths, or an iterator thereof.
        """
        if chunk_size is not None:
            return (
                self.trace_path_candidates(scene, cands, types)
                for cands, types in self.generate_path_candidates_chunks_iter(
                    scene, order, chunk_size=chunk_size, pad_chunks=pad_chunks
                )
            )
        candidates, interactions = self.generate_path_candidates(scene, order)
        return self.trace_path_candidates(scene, candidates, interactions)


class AbstractPathLauncher(AbstractPathSolver):
    """Abstract base class for ray-launching path solvers.

    Subclasses must implement :meth:`launch_rays` and may override
    :meth:`bounce_rays` and :meth:`filter_rays`.

    The main entry point is :meth:`launch_paths`, which orchestrates
    ray launching, bouncing, filtering, and assembly; analogous to
    :meth:`AbstractPathTracer.trace_paths`.
    """

    max_dist: AbstractVar[float]
    """Maximal (squared) distance between a receiver and a ray."""

    @abc.abstractmethod
    def launch_rays(
        self,
        scene: "TriangleScene",
    ) -> tuple[Float[Array, "num_tx num_rays 3"], Float[Array, "num_tx num_rays 3"]]:
        """
        Launch rays from transmitters.

        Args:
            scene: The scene.

        Returns:
            A tuple of ray origins and ray directions.
        """

    def bounce_rays(  # noqa: PLR6301
        self,
        scene: "TriangleScene",
        ray_origins: Float[Array, "num_tx num_rays 3"],
        ray_directions: Float[Array, "num_tx num_rays 3"],
        triangles: Int[Array, "num_tx num_rays"],
        t_hit: Float[Array, "num_tx num_rays"],
        valid_rays: Bool[Array, "num_tx num_rays"],
    ) -> tuple[
        Float[Array, "num_tx num_rays 3"],
        Float[Array, "num_tx num_rays 3"],
        Bool[Array, "num_tx num_rays"],
    ]:
        """
        Apply ray bouncing strategies to update ray states.

        Args:
            scene: The scene.
            ray_origins: The current ray origins.
            ray_directions: The current ray directions.
            triangles: The hit triangles.
            t_hit: The distance to hit.
            valid_rays: The boolean mask indicating valid rays.

        Returns:
            A tuple of updated ray origins, updated ray directions, and updated valid mask.
        """
        inside_scene = jnp.isfinite(t_hit)
        valid_rays = valid_rays & inside_scene
        t_hit = jnp.where(inside_scene, t_hit, jnp.zeros_like(t_hit))

        ray_origins = ray_origins + t_hit[..., None] * ray_directions
        mirror_normals = jnp.take(scene.mesh.normals, triangles, axis=0)
        ray_directions = (
            ray_directions
            - 2.0
            * jnp.sum(ray_directions * mirror_normals, axis=-1, keepdims=True)
            * mirror_normals
        )
        return ray_origins, ray_directions, valid_rays

    def filter_rays(
        self,
        scene: "TriangleScene",  # noqa: ARG002
        ray_origins: Float[Array, "num_tx num_rays 3"],
        ray_directions: Float[Array, "num_tx num_rays 3"],
        rx_vertices: Float[Array, "num_rx 3"],
        t_hit: Float[Array, "num_tx num_rays"],
        valid_rays: Bool[Array, "num_tx num_rays"],
    ) -> Bool[Array, "num_tx num_rx num_rays"]:
        """
        Filter rays by some criteria around receiver positions.

        Args:
            scene: The scene.
            ray_origins: The ray origins at start of bounce segment.
            ray_directions: The ray directions.
            rx_vertices: The receiver positions.
            t_hit: The distance to hit (end of bounce segment).
            valid_rays: The boolean mask indicating valid rays.

        Returns:
            A boolean mask indicating which rays pass near which receivers.
        """
        ray_origins_to_rx_vertices = (
            rx_vertices[None, :, None, :] - ray_origins[:, None, ...]
        )
        ray_distances_to_rx_vertices = jnp.square(
            jnp.cross(ray_directions[:, None, ...], ray_origins_to_rx_vertices)
        ).sum(axis=-1)
        t_rxs = jnp.sum(
            ray_directions[:, None, ...] * ray_origins_to_rx_vertices, axis=-1
        )
        return jnp.where(
            (t_rxs > 0) & (t_rxs < t_hit[:, None, :]) & valid_rays[:, None, :],
            ray_distances_to_rx_vertices < self.max_dist,
            False,
        )

    @eqx.filter_jit
    def launch_paths(
        self,
        scene: "TriangleScene",
        order: int,
    ) -> LaunchedPaths:
        """Launch paths for the given scene and order.

        Orchestrates :meth:`launch_rays`, :meth:`bounce_rays`, and
        :meth:`filter_rays` into a complete
        :class:`~differt.geometry.LaunchedPaths` result.

        Args:
            scene: The scene.
            order: The maximum path order (number of bounces).

        Returns:
            The launched paths.
        """
        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)

        ray_origins, ray_directions = self.launch_rays(scene)
        num_tx_vertices = tx_vertices.shape[0]
        num_rx_vertices = rx_vertices.shape[0]
        num_rays = ray_origins.shape[1]

        def scan_fun(
            ray_origins_directions_and_valids: tuple[
                Float[Array, "num_tx num_rays 3"],
                Float[Array, "num_tx num_rays 3"],
                Bool[Array, "num_tx num_rays"],
            ],
            _: None,
        ) -> tuple[
            tuple[
                Float[Array, "num_tx num_rays 3"],
                Float[Array, "num_tx num_rays 3"],
                Bool[Array, "num_tx num_rays"],
            ],
            tuple[
                Int[Array, "num_tx num_rays"],
                Float[Array, "num_tx num_rays 3"],
                Bool[Array, "num_tx num_rx num_rays"],
            ],
        ]:
            (
                ray_origins,
                ray_directions,
                valid_rays,
            ) = ray_origins_directions_and_valids

            triangles, t_hit = scene.mesh.first_triangle_hit_by_ray(
                ray_origins,
                ray_directions,
            )

            masks = self.filter_rays(
                scene,
                ray_origins,
                ray_directions,
                rx_vertices,
                t_hit,
                valid_rays,
            )

            ray_origins, ray_directions, valid_rays = self.bounce_rays(
                scene,
                ray_origins,
                ray_directions,
                triangles,
                t_hit,
                valid_rays,
            )

            return (ray_origins, ray_directions, valid_rays), (
                triangles,
                ray_origins,
                masks,
            )

        valid_rays = jnp.ones(ray_origins.shape[:-1], dtype=bool)
        _, (path_candidates, vertices, masks) = jax.lax.scan(
            scan_fun,
            (ray_origins, ray_directions, valid_rays),
            length=order + 1,
        )

        path_candidates = jnp.moveaxis(path_candidates[:-1, ...], 0, -1)
        vertices = jnp.moveaxis(vertices[:-1, ...], 0, -2)
        masks = jnp.moveaxis(masks, 0, -1)

        vertices = assemble_path(
            tx_vertices[:, None, None, :],
            vertices[:, None, ...],
            rx_vertices[None, :, None, :],
        )

        object_dtype = path_candidates.dtype

        tx_objects = jnp.arange(num_tx_vertices, dtype=object_dtype)
        rx_objects = jnp.arange(num_rx_vertices, dtype=object_dtype)

        tx_objects = jnp.broadcast_to(
            tx_objects[:, None, None, None],
            (num_tx_vertices, num_rx_vertices, num_rays, 1),
        )
        rx_objects = jnp.broadcast_to(
            rx_objects[None, :, None, None],
            (num_tx_vertices, num_rx_vertices, num_rays, 1),
        )
        path_candidates = jnp.broadcast_to(
            path_candidates[:, None, ...],
            (
                num_tx_vertices,
                num_rx_vertices,
                num_rays,
                order,
            ),
        )

        objects = jnp.concatenate((tx_objects, path_candidates, rx_objects), axis=-1)

        # All bounces are specular reflections (value 0) for SBR
        interaction_types = jnp.zeros(
            (num_tx_vertices, num_rx_vertices, num_rays, order), dtype=jnp.int32
        )

        return LaunchedPaths(
            vertices=vertices,
            objects=objects,
            masks=masks,
            interaction_types=interaction_types,
        )


# ---------------------------------------------------------------------------
# Helper: trace path candidates (image method + validation)
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _trace_path_candidates(
    mesh: TriangleMesh,
    tx_vertices: Float[Array, "num_tx_vertices 3"],
    rx_vertices: Float[Array, "num_rx_vertices 3"],
    path_candidates: Int[Array, "num_path_candidates order"],
    interaction_types: Int[Array, "num_path_candidates order"] | None = None,
    *,
    epsilon: Float[ArrayLike, ""] | None,
    hit_tol: Float[ArrayLike, ""] | None,
    min_len: Float[ArrayLike, ""] | None,
    smoothing_factor: Float[ArrayLike, ""] | None,
    confidence_threshold: Float[ArrayLike, ""],
    batch_size: int | None,
) -> TracedPaths:
    if min_len is None:
        dtype = jnp.result_type(mesh.vertices, tx_vertices, rx_vertices)
        min_len = 10.0 * jnp.finfo(dtype).eps

    min_len = jnp.asarray(min_len)

    # 1 - Broadcast arrays

    num_tx_vertices = tx_vertices.shape[0]
    num_rx_vertices = rx_vertices.shape[0]
    num_path_candidates, order = path_candidates.shape

    if mesh.assume_quads:
        # [num_path_candidates 2*order]
        path_candidates = jnp.repeat(path_candidates, 2, axis=-1)
        path_candidates = path_candidates.at[..., 1::2].add(1)  # Shift odd indices by 1
        k = 2
    else:
        k = 1

    # [num_path_candidates k*order 3]
    triangles = jnp.take(mesh.triangles, path_candidates, axis=0).reshape(
        num_path_candidates, k * order, 3
    )  # reshape required if mesh is empty

    # [num_path_candidates k*order 3 3]
    triangle_vertices = jnp.take(mesh.vertices, triangles, axis=0).reshape(
        num_path_candidates, k * order, 3, 3
    )  # reshape required if mesh is empty

    if mesh.mask is not None:
        # For a ray to be active, it must hit triangles that are not masked out (i.e, inactive).
        # [num_path_candidates]
        active_rays = jnp.take(mesh.mask, path_candidates, axis=0).all(axis=-1)
    else:
        active_rays = None

    # [num_path_candidates order 3]
    mirror_vertices = triangle_vertices[
        ...,
        :: (2 if mesh.assume_quads else 1),
        0,
        :,
    ]  # Only one vertex per triangle is needed

    # [num_path_candidates order 3]
    mirror_normals = jnp.take(
        mesh.normals, path_candidates[..., :: (2 if mesh.assume_quads else 1)], axis=0
    )

    # 2 - Trace paths

    if num_path_candidates == 0:
        dtype = jnp.result_type(
            tx_vertices, rx_vertices, mirror_vertices, mesh.vertices
        )
        # [num_tx_vertices num_rx_vertices num_path_candidates order+2 3]
        full_paths = jnp.empty(
            (num_tx_vertices, num_rx_vertices, 0, order + 2, 3), dtype=dtype
        )
    else:
        # [num_tx_vertices num_rx_vertices num_path_candidates order 3]
        paths = image_method(
            tx_vertices[:, None, None, :],
            rx_vertices[None, :, None, :],
            mirror_vertices,
            mirror_normals,
        )
        full_paths = assemble_path(
            tx_vertices[:, None, None, :],
            paths,
            rx_vertices[None, :, None, :],
        )

    # 3 - Identify invalid paths

    # [num_tx_vertices num_rx_vertices num_path_candidates order+1 3]
    ray_origins = full_paths[..., :-1, :]
    # [num_tx_vertices num_rx_vertices num_path_candidates order+1 3]
    ray_directions = jnp.diff(full_paths, axis=-2)

    # 3.1 - Check if paths vertices are inside respective triangles

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if mesh.assume_quads:
        if smoothing_factor is not None:
            inside_triangles = (
                ray_intersect_triangle(
                    jnp.repeat(ray_origins[..., :-1, :], 2, axis=-2),
                    jnp.repeat(ray_directions[..., :-1, :], 2, axis=-2),
                    triangle_vertices,
                    epsilon=epsilon,
                    smoothing_factor=smoothing_factor,
                )[1]
                .reshape(
                    num_tx_vertices, num_rx_vertices, num_path_candidates, order, 2
                )
                .max(axis=-1, initial=0.0)
                .min(axis=-1, initial=1.0)
            )  # Reduce on 'order' axis and on the two triangles (per quad)
        else:
            inside_triangles = (
                ray_intersect_triangle(
                    jnp.repeat(ray_origins[..., :-1, :], 2, axis=-2),
                    jnp.repeat(ray_directions[..., :-1, :], 2, axis=-2),
                    triangle_vertices,
                    epsilon=epsilon,
                )[1]
                .reshape(
                    num_tx_vertices, num_rx_vertices, num_path_candidates, order, 2
                )
                .any(axis=-1)
                .all(axis=-1)
            )  # Reduce on 'order' axis and on the two triangles (per quad)
    elif smoothing_factor is not None:
        inside_triangles = ray_intersect_triangle(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
            smoothing_factor=smoothing_factor,
        )[1].min(axis=-1, initial=1.0)  # Reduce on 'order' axis
    else:
        inside_triangles = ray_intersect_triangle(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
        )[1].all(axis=-1)  # Reduce on 'order' axis

    # 3.2 - Check if consecutive path vertices are on the same side of mirrors

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if smoothing_factor is not None:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirror(
            full_paths,
            mirror_vertices,
            mirror_normals,
            smoothing_factor=smoothing_factor,
        ).min(axis=-1, initial=1.0)  # Reduce on 'order'
    else:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirror(
            full_paths,
            mirror_vertices,
            mirror_normals,
        ).all(axis=-1)  # Reduce on 'order'

    # 3.3 - Identify paths that are blocked by other objects

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if smoothing_factor is not None:
        blocked = ray_intersect_any_triangle(
            ray_origins,
            ray_directions,
            mesh.triangle_vertices,
            active_triangles=mesh.mask,
            epsilon=epsilon,
            hit_tol=hit_tol,
            smoothing_factor=smoothing_factor,
            batch_size=batch_size,
        ).max(axis=-1, initial=0.0)  # Reduce on 'order'
    else:  # Use faster implementation
        blocked = mesh.ray_intersect_any_triangle(
            ray_origins,
            ray_directions,
            hit_tol=hit_tol,
        ).any(axis=-1)  # Reduce on 'order'

    # 3.4 - Identify path segments that are too small (e.g., double-reflection inside an edge)

    ray_lengths = jnp.sum(ray_directions * ray_directions, axis=-1)  # Squared norm

    if smoothing_factor is not None:
        too_small = smoothing_function(min_len - ray_lengths, smoothing_factor).max(
            axis=-1, initial=0.0
        )  # Any path segment being too small
    else:
        too_small = (ray_lengths < min_len).any(
            axis=-1
        )  # Any path segment being too small

    # 3.5 - Identify paths that are not finite
    is_finite = jnp.isfinite(full_paths).all(axis=(-1, -2))
    full_paths = jnp.where(
        is_finite[..., None, None], full_paths, jnp.zeros_like(full_paths)
    )

    if smoothing_factor is not None:
        mask = jnp.stack(
            (
                inside_triangles,
                valid_reflections,
                1.0 - blocked,
                1.0 - too_small,
                is_finite.astype(inside_triangles.dtype),
            ),
            axis=-1,
        ).min(axis=-1, initial=1.0)
        if active_rays is not None:
            mask *= active_rays
    else:
        mask = inside_triangles & valid_reflections & ~blocked & ~too_small & is_finite
        if active_rays is not None:
            mask &= active_rays

    vertices = full_paths

    # 4 - Generate output paths and reshape

    object_dtype = path_candidates.dtype

    tx_objects = jnp.arange(num_tx_vertices, dtype=object_dtype)
    rx_objects = jnp.arange(num_rx_vertices, dtype=object_dtype)

    tx_objects = jnp.broadcast_to(
        tx_objects[:, None, None, None],
        (num_tx_vertices, num_rx_vertices, num_path_candidates, 1),
    )
    rx_objects = jnp.broadcast_to(
        rx_objects[None, :, None, None],
        (num_tx_vertices, num_rx_vertices, num_path_candidates, 1),
    )
    path_candidates_for_objects = jnp.broadcast_to(
        path_candidates[:, ::k],
        (
            num_tx_vertices,
            num_rx_vertices,
            num_path_candidates,
            order,
        ),
    )

    objects = jnp.concatenate(
        (tx_objects, path_candidates_for_objects, rx_objects), axis=-1
    )

    # Build interaction_types for the output
    if interaction_types is not None:
        # Broadcast to match TX/RX dims
        out_interaction_types = jnp.broadcast_to(
            interaction_types,
            (num_tx_vertices, num_rx_vertices, num_path_candidates, order),
        )
    else:
        # Default: all specular reflections (value 0)
        out_interaction_types = jnp.zeros(
            (num_tx_vertices, num_rx_vertices, num_path_candidates, order),
            dtype=jnp.int32,
        )

    return TracedPaths(
        vertices,
        objects,
        mask=mask,
        interaction_types=out_interaction_types,
        confidence_threshold=confidence_threshold,
    )


# ---------------------------------------------------------------------------
# Concrete solvers
# ---------------------------------------------------------------------------


class ExhaustivePathTracer(AbstractPathTracer):
    """
    Exhaustive (image-method) path tracer.

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

    def generate_path_candidates(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        specular_reflection: bool = True,  # noqa: ARG002
        diffuse_scattering: bool = False,  # noqa: ARG002
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        if isinstance(order, Sequence):
            msg = "ExhaustivePathTracer does not support multiple orders yet."
            raise NotImplementedError(msg)

        graph = CompleteGraph(scene.mesh.num_primitives)
        assume_quads = scene.mesh.assume_quads

        if self.disconnect_inactive_triangles and scene.mesh.mask is not None:
            mask = scene.mesh.mask
            if assume_quads:
                mask = mask[0::2] & mask[1::2]

            graph = DiGraph.from_complete_graph(graph)
            from_, to = graph.insert_from_and_to_nodes()
            graph.filter_by_mask(np.asarray(mask), fast_mode=True)
        else:
            from_ = graph.num_nodes
            to = from_ + 1

        path_candidates = jnp.asarray(
            graph.all_paths_array(
                from_=from_,
                to=to,
                depth=order + 2,
                include_from_and_to=False,
            ),
            dtype=int,
        )

        if assume_quads:
            path_candidates = 2 * path_candidates

        # Default: all specular reflections (value 0)
        interaction_types = jnp.zeros_like(path_candidates, dtype=jnp.int32)

        return path_candidates, interaction_types

    def generate_path_candidates_chunks_iter(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        *args: Any,
        chunk_size: int | None = None,
        pad_chunks: bool = False,
        **kwargs: Any,
    ) -> SizedIterator[
        tuple[
            Int[Array, "... chunk_size order"],
            Int[Array, "... chunk_size order"],
        ]
    ]:
        """Override to support native chunked generation from the graph.

        Returns:
            An iterator over path candidates chunks.
        """
        # Use instance chunk_size if not explicitly provided
        effective_chunk_size = chunk_size or self.chunk_size
        if effective_chunk_size is None:
            # Fall back to generating all at once and wrapping as a single chunk
            candidates, interactions = self.generate_path_candidates(
                scene, order, *args, **kwargs
            )
            return SizedIterator(iter([(candidates, interactions)]), size=1)

        if isinstance(order, Sequence):
            msg = "ExhaustivePathTracer does not support multiple orders yet."
            raise NotImplementedError(msg)

        graph = CompleteGraph(scene.mesh.num_primitives)
        assume_quads = scene.mesh.assume_quads

        if self.disconnect_inactive_triangles and scene.mesh.mask is not None:
            mask = scene.mesh.mask
            if assume_quads:
                mask = mask[0::2] & mask[1::2]

            graph = DiGraph.from_complete_graph(graph)
            from_, to = graph.insert_from_and_to_nodes()
            graph.filter_by_mask(np.asarray(mask), fast_mode=True)
        else:
            from_ = graph.num_nodes
            to = from_ + 1

        path_candidates_iter = graph.all_paths_array_chunks(
            from_=from_,
            to=to,
            depth=order + 2,
            include_from_and_to=False,
            chunk_size=effective_chunk_size,
        )

        def gen() -> Iterator[
            tuple[
                Int[Array, "chunk_size order"],
                Int[Array, "chunk_size order"],
            ]
        ]:
            for chunk_arr in path_candidates_iter:
                if pad_chunks and len(chunk_arr) < effective_chunk_size:
                    pad_width = ((0, effective_chunk_size - len(chunk_arr)), (0, 0))
                    padded_chunk = np.pad(
                        chunk_arr, pad_width, mode="constant", constant_values=-1
                    )
                else:
                    padded_chunk = chunk_arr

                candidates_chunk = jnp.asarray(padded_chunk, dtype=int)
                if assume_quads:
                    candidates_chunk = 2 * candidates_chunk
                interaction_types_chunk = jnp.zeros_like(
                    candidates_chunk, dtype=jnp.int32
                )
                yield candidates_chunk, interaction_types_chunk

        if hasattr(path_candidates_iter, "__len__"):
            size: int | Callable[[], int] = path_candidates_iter.__len__
        else:
            # Cannot know size ahead of time
            size = -1

        return SizedIterator(gen(), size=size)

    @eqx.filter_jit
    def trace_path_candidates(
        self,
        scene: "TriangleScene",
        path_candidates: Int[Array, "num_candidates order"],
        interaction_types: Int[Array, "num_candidates order"],
    ) -> TracedPaths:
        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)
        return _trace_path_candidates(
            scene.mesh,
            tx_vertices,
            rx_vertices,
            path_candidates,
            interaction_types=interaction_types,
            epsilon=self.epsilon,
            hit_tol=self.hit_tol,
            min_len=self.min_len,
            smoothing_factor=self.smoothing_factor,
            confidence_threshold=self.confidence_threshold,
            batch_size=self.batch_size,
        )


class HybridPathTracer(AbstractPathTracer):
    """
    Hybrid path tracer, combining ray launching for visibility and exhaustive tracing.

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

    def generate_path_candidates(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        specular_reflection: bool = True,  # noqa: ARG002
        diffuse_scattering: bool = False,  # noqa: ARG002
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        if isinstance(order, Sequence):
            msg = "HybridPathTracer does not support multiple orders yet."
            raise NotImplementedError(msg)

        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)

        assume_quads = scene.mesh.assume_quads
        graph = CompleteGraph(scene.mesh.num_primitives)

        triangles_visible_from_tx = scene.mesh.triangles_visible_from_vertex(
            tx_vertices,
            num_rays=self.num_rays,
        ).any(axis=0)

        triangles_visible_from_rx = scene.mesh.triangles_visible_from_vertex(
            rx_vertices,
            num_rays=self.num_rays,
        ).any(axis=0)

        if assume_quads:
            triangles_visible_from_tx = triangles_visible_from_tx.reshape(-1, 2).any(
                axis=-1
            )
            triangles_visible_from_rx = triangles_visible_from_rx.reshape(-1, 2).any(
                axis=-1
            )

        graph = DiGraph.from_complete_graph(graph)
        from_, to = graph.insert_from_and_to_nodes(
            from_adjacency=np.asarray(triangles_visible_from_tx),
            to_adjacency=np.asarray(triangles_visible_from_rx),
        )
        if scene.mesh.mask is not None:
            mask = scene.mesh.mask
            if assume_quads:
                mask = mask[0::2] & mask[1::2]
            graph.filter_by_mask(np.asarray(mask), fast_mode=True)

        path_candidates = jnp.asarray(
            graph.all_paths_array(
                from_=from_,
                to=to,
                depth=order + 2,
                include_from_and_to=False,
            ),
            dtype=int,
        )

        if assume_quads:
            path_candidates = 2 * path_candidates

        # Default: all specular reflections (value 0)
        interaction_types = jnp.zeros_like(path_candidates, dtype=jnp.int32)

        return path_candidates, interaction_types

    def generate_path_candidates_chunks_iter(
        self,
        scene: "TriangleScene",
        order: int | Sequence[int],
        *args: Any,
        chunk_size: int | None = None,
        pad_chunks: bool = False,  # noqa: ARG002
        **kwargs: Any,
    ) -> SizedIterator[
        tuple[
            Int[Array, "... chunk_size order"],
            Int[Array, "... chunk_size order"],
        ]
    ]:
        """Override to support native chunked generation from the graph.

        Returns:
            An iterator over path candidates chunks.
        """
        effective_chunk_size = chunk_size or self.chunk_size
        if effective_chunk_size is None:
            candidates, interactions = self.generate_path_candidates(
                scene, order, *args, **kwargs
            )
            return SizedIterator(iter([(candidates, interactions)]), size=1)

        if isinstance(order, Sequence):
            msg = "HybridPathTracer does not support multiple orders yet."
            raise NotImplementedError(msg)

        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)

        assume_quads = scene.mesh.assume_quads
        graph = CompleteGraph(scene.mesh.num_primitives)

        triangles_visible_from_tx = scene.mesh.triangles_visible_from_vertex(
            tx_vertices,
            num_rays=self.num_rays,
        ).any(axis=0)

        triangles_visible_from_rx = scene.mesh.triangles_visible_from_vertex(
            rx_vertices,
            num_rays=self.num_rays,
        ).any(axis=0)

        if assume_quads:
            triangles_visible_from_tx = triangles_visible_from_tx.reshape(-1, 2).any(
                axis=-1
            )
            triangles_visible_from_rx = triangles_visible_from_rx.reshape(-1, 2).any(
                axis=-1
            )

        graph = DiGraph.from_complete_graph(graph)
        from_, to = graph.insert_from_and_to_nodes(
            from_adjacency=np.asarray(triangles_visible_from_tx),
            to_adjacency=np.asarray(triangles_visible_from_rx),
        )
        if scene.mesh.mask is not None:
            mask = scene.mesh.mask
            if assume_quads:
                mask = mask[0::2] & mask[1::2]
            graph.filter_by_mask(np.asarray(mask), fast_mode=True)

        path_candidates_iter = graph.all_paths_array_chunks(
            from_=from_,
            to=to,
            depth=order + 2,
            include_from_and_to=False,
            chunk_size=effective_chunk_size,
        )

        def gen() -> Iterator[
            tuple[
                Int[Array, "chunk_size order"],
                Int[Array, "chunk_size order"],
            ]
        ]:
            for chunk_arr in path_candidates_iter:
                candidates_chunk = jnp.asarray(chunk_arr, dtype=int)
                if assume_quads:
                    candidates_chunk = 2 * candidates_chunk
                interaction_types_chunk = jnp.zeros_like(
                    candidates_chunk, dtype=jnp.int32
                )
                yield candidates_chunk, interaction_types_chunk

        if hasattr(path_candidates_iter, "__len__"):
            size: int | Callable[[], int] = path_candidates_iter.__len__
        else:
            size = -1

        return SizedIterator(gen(), size=size)

    @eqx.filter_jit
    def trace_path_candidates(
        self,
        scene: "TriangleScene",
        path_candidates: Int[Array, "num_candidates order"],
        interaction_types: Int[Array, "num_candidates order"],
    ) -> TracedPaths:
        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)
        return _trace_path_candidates(
            scene.mesh,
            tx_vertices,
            rx_vertices,
            path_candidates,
            interaction_types=interaction_types,
            epsilon=self.epsilon,
            hit_tol=self.hit_tol,
            min_len=self.min_len,
            smoothing_factor=self.smoothing_factor,
            confidence_threshold=self.confidence_threshold,
            batch_size=self.batch_size,
        )


class SBRPathLauncher(AbstractPathLauncher):
    """
    Shooting-and-bouncing ray (SBR) path launcher.

    A fixed number of rays are launched from each transmitter and are allowed
    to perform a fixed number of bounces. Only ray paths passing in the vicinity
    of a receiver are considered valid.

    .. important::

        This SBR method is currently unstable and not yet optimized, and it is likely
        to change in future releases. Use with caution.
    """

    num_rays: int = int(1e6)
    """The number of rays launched."""
    epsilon: Float[ArrayLike, ""] | None = None
    """Tolerance for checking ray / object intersections."""
    hit_tol: Float[ArrayLike, ""] | None = None
    """Tolerance for blockage checks."""
    max_dist: Float[ArrayLike, ""] = 1e-3
    """Maximal (squared) distance between a receiver and a ray for the receiver to be considered in the vicinity of the ray path."""

    def launch_rays(
        self,
        scene: "TriangleScene",
    ) -> tuple[Float[Array, "num_tx num_rays 3"], Float[Array, "num_tx num_rays 3"]]:
        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)
        num_tx_vertices = tx_vertices.shape[0]
        triangle_vertices = scene.mesh.triangle_vertices

        world_vertices = jnp.concatenate(
            (triangle_vertices.reshape(-1, 3), rx_vertices), axis=0
        )

        frustums = jax.vmap(viewing_frustum, in_axes=(0, None))(
            tx_vertices, world_vertices
        )

        ray_origins = jnp.broadcast_to(
            tx_vertices[:, None, :], (num_tx_vertices, self.num_rays, 3)
        )
        ray_directions = jax.vmap(
            lambda frustum: fibonacci_lattice(self.num_rays, frustum=frustum)
        )(frustums)

        return ray_origins, ray_directions


class _ExhaustivePathTracerKwargs(TypedDict, total=False):
    epsilon: Float[ArrayLike, ""] | None
    hit_tol: Float[ArrayLike, ""] | None
    min_len: Float[ArrayLike, ""] | None
    smoothing_factor: Float[ArrayLike, ""] | None
    confidence_threshold: Float[ArrayLike, ""]
    batch_size: int | None
    disconnect_inactive_triangles: bool
    chunk_size: int | None


class _HybridPathTracerKwargs(TypedDict, total=False):
    num_rays: int
    epsilon: Float[ArrayLike, ""] | None
    hit_tol: Float[ArrayLike, ""] | None
    min_len: Float[ArrayLike, ""] | None
    smoothing_factor: Float[ArrayLike, ""] | None
    confidence_threshold: Float[ArrayLike, ""]
    batch_size: int | None
    chunk_size: int | None


class _SBRPathLauncherKwargs(TypedDict, total=False):
    num_rays: int
    epsilon: Float[ArrayLike, ""] | None
    hit_tol: Float[ArrayLike, ""] | None
    max_dist: Float[ArrayLike, ""]


__all__ = [
    "_ExhaustivePathTracerKwargs",
    "_HybridPathTracerKwargs",
    "_SBRPathLauncherKwargs",
]

import abc
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, no_type_check, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from equinox import AbstractVar
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.utils import smoothing_function
from differt_core.geometry import CompleteGraph, DiGraph

from ._mesh import Mesh
from ._paths import LaunchedPaths, TracedPaths
from ._solver_image_method import (
    consecutive_vertices_are_on_same_side_of_mirror,
    image_method,
)
from ._utils import (
    SizedIterator,
    assemble_path,
    check_path_candidates,
    fibonacci_lattice,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
    viewing_frustum,
)
from ._warp_utils import _Batched, _get_warp_mesh, _warp_launch

if TYPE_CHECKING:
    from ._scene import Scene


def _normalize_order(order: int | Sequence[int] | slice) -> Sequence[int]:
    """Normalize an ``order`` argument into a sequence of (non-negative) orders.

    A bare ``int`` is wrapped into a single-element tuple, and a ``slice``
    is converted into an equivalent :class:`range`; a :class:`~collections.abc.Sequence`
    (e.g., a :class:`range`) is left untouched. This way, callers never need
    to special-case whether ``order`` was given as a single order or as a
    sequence thereof.

    Args:
        order: The order, as accepted by path solver methods.

    Returns:
        The order(s), always as a :class:`~collections.abc.Sequence` of
        non-negative integers.

    Raises:
        ValueError: If ``order`` is a ``slice`` with an undefined ``stop``,
            or if any order is negative.
    """
    if isinstance(order, int):
        order = (order,)
    elif isinstance(order, slice):
        if order.stop is None:
            msg = (
                "A 'slice' order must have a defined 'stop', "
                "e.g., 'slice(0, 6)' or 'slice(None, 6)'."
            )
            raise ValueError(msg)
        order = range(order.start or 0, order.stop, order.step or 1)

    if any(o < 0 for o in order):
        msg = f"Order(s) must be non-negative, got {order!r}."
        raise ValueError(msg)

    return order


class AbstractPathSolver(eqx.Module):
    """Abstract base class for all path solvers and launchers.

    Subclasses should define concrete values for
    ``epsilon`` and ``hit_tol``.
    """

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
        scene: "Scene",
        order: int | Sequence[int] | slice,
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
        or padded bounce.

        ``order`` may also be a sequence of orders, e.g., ``[1, 2, 3]``, a
        :class:`range` (e.g., ``range(0, 6)``), or a ``slice`` with a
        defined ``stop`` (e.g., ``slice(0, 6)``, equivalent to
        ``range(0, 6)``), to combine candidates of multiple orders into a
        single array, with lower-order candidates padded with ``-1`` up to
        the maximum requested order, see :func:`check_path_candidates
        <differt.geometry.check_path_candidates>` for the exact placeholder
        convention. For most solvers, candidates are generated independently
        for each order and concatenated, so the size of the returned arrays
        is known ahead of time: it is the sum of the number of candidates
        generated for each individual order. :class:`SBRPathTracer` is a
        notable exception: it shares a single, fixed-size buffer across all
        requested orders instead, see its documentation for details.

        Args:
            scene: The scene.
            order: The path order (number of bounces), or a sequence of
                orders (also accepted as a :class:`range` or ``slice``) to
                combine.
            specular_reflection: Whether to include specular reflections.
            diffuse_scattering: Whether to include diffuse scattering
                (not yet implemented).

        Returns:
            A 2-tuple of ``(path_candidates, interaction_types)``.
        """

    def generate_path_candidates_chunks_iter(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
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
            A :class:`~differt.geometry.SizedIterator` over
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

    def _single_chunk_fallback(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
        *args: Any,
        chunk_size: int | None,
        **kwargs: Any,
    ) -> (
        SizedIterator[
            tuple[
                Int[Array, "... num_path_candidates max_order"],
                Int[Array, "... num_path_candidates max_order"],
            ]
        ]
        | None
    ):
        """Fall back to a single, unchunked chunk when no ``chunk_size`` is set.

        Shared by :class:`ExhaustivePathTracer`, :class:`HybridPathTracer`, and
        :class:`SBRPathTracer`'s ``generate_path_candidates_chunks_iter`` overrides:
        each first resolves its own effective chunk size (e.g., ``chunk_size or
        self.chunk_size``) and calls this helper with that value, returning its
        result directly whenever it is not :data:`None`; otherwise, native
        chunked generation proceeds using the (necessarily non-``None``)
        effective chunk size.

        Args:
            scene: The scene.
            order: The path order(s).
            *args: Forwarded to :meth:`generate_path_candidates`.
            chunk_size: The caller's already-resolved effective chunk size.
            **kwargs: Forwarded to :meth:`generate_path_candidates`.

        Returns:
            A :class:`~differt.geometry.SizedIterator` wrapping a single chunk
            with all path candidates, if ``chunk_size`` is :data:`None`;
            otherwise :data:`None`.
        """
        if chunk_size is not None:
            return None

        candidates, interactions = self.generate_path_candidates(
            scene, order, *args, **kwargs
        )
        return SizedIterator(iter([(candidates, interactions)]), size=1)

    @abc.abstractmethod
    def trace_path_candidates(
        self,
        scene: "Scene",
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
        scene: "Scene",
        order: int | Sequence[int] | slice,
        chunk_size: None = None,
        pad_chunks: bool = False,
    ) -> TracedPaths: ...

    @overload
    def trace_paths(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
        chunk_size: int,
        pad_chunks: bool = False,
    ) -> SizedIterator[TracedPaths]: ...

    def trace_paths(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
        chunk_size: int | None = None,
        pad_chunks: bool = False,
    ) -> TracedPaths | SizedIterator[TracedPaths]:
        """
        Trace paths for the given scene and order(s).

        If ``chunk_size`` is provided, returns an iterator of
        :class:`~differt.geometry.TracedPaths` (one per chunk);
        otherwise returns a single :class:`~differt.geometry.TracedPaths`.

        If ``order`` is a sequence of orders, e.g., ``[1, 2, 3]``, then
        :meth:`generate_path_candidates` directly generates path candidates
        for every requested order, combining them into a single array, with
        lower-order candidates padded (with ``-1``) up to the maximum
        requested order; the (single) combined array is then traced in one
        call to :meth:`trace_path_candidates`. This is not compatible with
        ``chunk_size``.

        Args:
            scene: The scene.
            order: The path order(s).
            chunk_size: If not ``None``, iterate through chunks of
                this size.
            pad_chunks: If ``True`` and ``chunk_size`` is set,
                pad the last chunk.

        Returns:
            Traced paths, or a sized iterator thereof.

        Raises:
            NotImplementedError: If ``order`` is a sequence of orders and
                ``chunk_size`` is not :data:`None`.
        """
        if not isinstance(order, int) and chunk_size is not None:
            msg = "Chunked generation ('chunk_size') is not supported when 'order' is a sequence of orders."  # TODO: implement me, this should be relatively easy to do
            raise NotImplementedError(msg)

        if chunk_size is not None:
            chunks_iter = self.generate_path_candidates_chunks_iter(
                scene, order, chunk_size=chunk_size, pad_chunks=pad_chunks
            )
            return SizedIterator(
                (
                    self.trace_path_candidates(scene, cands, types)
                    for cands, types in chunks_iter
                ),
                size=chunks_iter.__len__,
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
        scene: "Scene",
    ) -> tuple[Float[Array, "num_tx num_rays 3"], Float[Array, "num_tx num_rays 3"]]:
        """
        Launch rays from transmitters.

        Args:
            scene: The scene.

        Returns:
            A tuple of ray origins and ray directions.
        """

    def bounce_rays(  # ruff:ignore[no-self-use]
        self,
        scene: "Scene",
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
        scene: "Scene",  # ruff:ignore[unused-method-argument]
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
        scene: "Scene",
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


@eqx.filter_jit
def _trace_path_candidates(
    mesh: Mesh,
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

    # 0 - Validate path candidates and identify placeholder ('-1') interactions

    path_candidates = check_path_candidates(path_candidates)

    num_tx_vertices = tx_vertices.shape[0]
    num_rx_vertices = rx_vertices.shape[0]
    num_path_candidates, order = path_candidates.shape

    # [num_path_candidates order] - 'True' for genuine interactions, 'False' for
    # placeholder ('-1') ones, used to pad path candidates of a lower order.
    active = path_candidates >= 0
    # [num_path_candidates order] - like 'path_candidates', but with placeholder
    # values replaced by a valid (dummy) index, so it is always safe to use
    # for array indexing. Because placeholder positions are excluded below
    # from every validity criterion, which dummy primitive is used does not
    # affect the result.
    safe_path_candidates = jnp.where(active, path_candidates, 0)

    # 1 - Broadcast arrays

    if mesh.assume_quads:
        # [num_path_candidates 2*order]
        quad_path_candidates = jnp.repeat(safe_path_candidates, 2, axis=-1)
        # Shift odd indices by 1
        quad_path_candidates = quad_path_candidates.at[..., 1::2].add(1)
        k = 2
    else:
        quad_path_candidates = safe_path_candidates
        k = 1

    # [num_path_candidates k*order 3]
    triangles = jnp.take(mesh.triangles, quad_path_candidates, axis=0).reshape(
        num_path_candidates, k * order, 3
    )  # reshape required if mesh is empty

    # [num_path_candidates k*order 3 3]
    triangle_vertices = jnp.take(mesh.vertices, triangles, axis=0).reshape(
        num_path_candidates, k * order, 3, 3
    )  # reshape required if mesh is empty

    if mesh.mask is not None:
        # For a ray to be active, it must hit triangles that are not masked out (i.e, inactive).
        # Placeholder interactions are excluded from this requirement.
        # [num_path_candidates]
        active_rays = (jnp.take(mesh.mask, safe_path_candidates, axis=0) | ~active).all(
            axis=-1
        )
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
    mirror_normals = jnp.take(mesh.normals, safe_path_candidates, axis=0)

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
        # Placeholder interactions must not contribute any actual reflection.
        # We replace their mirror by the (receiver-dependent) infinite plane
        # that goes through the receiver: because the receiver trivially lies
        # on that plane, both the image (forward pass) and the intersection
        # point (backward pass) of the image method recursion collapse to the
        # receiver itself, for every placeholder position. As placeholders
        # only ever appear as a trailing suffix (checked above), this exactly
        # produces a path that reaches the receiver after its last genuine
        # interaction, then stays there for the remaining (padded) positions.
        # [num_rx_vertices num_path_candidates order 3]
        mirror_vertices_for_image_method = jnp.where(
            active[None, ..., None],
            mirror_vertices[None, ...],
            rx_vertices[:, None, None, :],
        )
        mirror_normals_for_image_method = jnp.where(
            active[None, ..., None],
            mirror_normals[None, ...],
            jnp.zeros_like(mirror_normals)[None, ...],
        )

        # [num_tx_vertices num_rx_vertices num_path_candidates order 3]
        paths = image_method(
            tx_vertices[:, None, None, :],
            rx_vertices[None, :, None, :],
            mirror_vertices_for_image_method,
            mirror_normals_for_image_method,
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

    # A path segment is genuine if it starts from the transmitter or from a
    # genuine interaction (the segment reaching the receiver right after the
    # last genuine interaction is also genuine); segments in between two
    # placeholder interactions are not, and must not affect validity.
    # [num_path_candidates order+1]
    segment_active = jnp.concatenate(
        (jnp.ones((num_path_candidates, 1), dtype=bool), active), axis=-1
    )

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
            )  # Reduce on the two triangles (per quad)
            inside_triangles = jnp.where(active, inside_triangles, 1.0).min(
                axis=-1, initial=1.0
            )  # Reduce on 'order' axis, ignoring placeholder interactions
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
            )  # Reduce on the two triangles (per quad)
            inside_triangles = (inside_triangles | ~active).all(
                axis=-1
            )  # Reduce on 'order' axis, ignoring placeholder interactions
    elif smoothing_factor is not None:
        inside_triangles = ray_intersect_triangle(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
            smoothing_factor=smoothing_factor,
        )[1]
        inside_triangles = jnp.where(active, inside_triangles, 1.0).min(
            axis=-1, initial=1.0
        )  # Reduce on 'order' axis, ignoring placeholder interactions
    else:
        inside_triangles = ray_intersect_triangle(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
        )[1]
        inside_triangles = (inside_triangles | ~active).all(
            axis=-1
        )  # Reduce on 'order' axis, ignoring placeholder interactions

    # 3.2 - Check if consecutive path vertices are on the same side of mirrors

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if smoothing_factor is not None:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirror(
            full_paths,
            mirror_vertices,
            mirror_normals,
            smoothing_factor=smoothing_factor,
        )
        valid_reflections = jnp.where(active, valid_reflections, 1.0).min(
            axis=-1, initial=1.0
        )  # Reduce on 'order', ignoring placeholder interactions
    else:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirror(
            full_paths,
            mirror_vertices,
            mirror_normals,
        )
        valid_reflections = (valid_reflections | ~active).all(
            axis=-1
        )  # Reduce on 'order', ignoring placeholder interactions

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
        )
        blocked = jnp.where(segment_active, blocked, 0.0).max(
            axis=-1, initial=0.0
        )  # Reduce on segments, ignoring non-genuine ones
    else:  # Use faster implementation
        blocked = mesh.ray_intersect_any_triangle(
            ray_origins,
            ray_directions,
            hit_tol=hit_tol,
        )
        blocked = (blocked & segment_active).any(
            axis=-1
        )  # Reduce on segments, ignoring non-genuine ones

    # 3.4 - Identify path segments that are too small (e.g., double-reflection inside an edge)

    ray_lengths = jnp.sum(ray_directions * ray_directions, axis=-1)  # Squared norm

    if smoothing_factor is not None:
        too_small = smoothing_function(min_len - ray_lengths, smoothing_factor)
        too_small = jnp.where(segment_active, too_small, 0.0).max(
            axis=-1, initial=0.0
        )  # Any genuine path segment being too small
    else:
        too_small = ray_lengths < min_len
        too_small = (too_small & segment_active).any(
            axis=-1
        )  # Any genuine path segment being too small

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
        path_candidates,
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


def _pad_path_candidates(
    path_candidates: Int[Array, "num_path_candidates order"],
    interaction_types: Int[Array, "num_path_candidates order"],
    order: int,
) -> tuple[
    Int[Array, "num_path_candidates {order}"], Int[Array, "num_path_candidates {order}"]
]:
    """Pad path candidates of a lower order up to ``order`` with placeholders.

    Args:
        path_candidates: The path candidates to pad.
        interaction_types: The corresponding interaction types.
        order: The target order, which must be greater than or equal to the
            current order (``path_candidates.shape[-1]``).

    Returns:
        The padded path candidates and interaction types.

    Raises:
        ValueError: If ``order`` is smaller than the current order.
    """
    current_order = path_candidates.shape[-1]
    missing = order - current_order

    if missing == 0:
        return path_candidates, interaction_types
    if missing < 0:
        msg = f"Cannot pad path candidates of order {current_order} down to a lower order {order}."
        raise ValueError(msg)

    pad_width = ((0, 0), (0, missing))
    path_candidates = jnp.pad(path_candidates, pad_width, constant_values=-1)
    interaction_types = jnp.pad(interaction_types, pad_width, constant_values=-1)

    return path_candidates, interaction_types


def _generate_path_candidates_for_orders(
    solver: "ExhaustivePathTracer | HybridPathTracer",
    scene: "Scene",
    orders: Sequence[int],
    specular_reflection: bool,
    diffuse_scattering: bool,
) -> tuple[
    Int[Array, "num_path_candidates max_order"],
    Int[Array, "num_path_candidates max_order"],
]:
    """Generate path candidates independently for each order, then combine them.

    Each order is generated with its own, unpadded call to
    ``solver._generate_path_candidates_for_one_order``. The resulting
    candidates are then padded (see :func:`_pad_path_candidates`) up to the
    maximum requested order, and concatenated into a single array, whose
    size is known ahead of time (the sum of the number of candidates
    generated for each individual order).

    Args:
        solver: The path tracer used to generate candidates for each order.
        scene: The scene.
        orders: The (non-empty) sequence of orders to generate and combine.
        specular_reflection: Whether to include specular reflections.
        diffuse_scattering: Whether to include diffuse scattering.

    Returns:
        The combined path candidates and interaction types.

    Raises:
        ValueError: If ``orders`` is empty.
    """
    order_list = sorted({int(o) for o in orders})

    if not order_list:
        msg = "You must provide at least one order when 'order' is a sequence."
        raise ValueError(msg)

    max_order = order_list[-1]

    candidates_and_types = [
        _pad_path_candidates(
            *solver._generate_path_candidates_for_one_order(  # ruff:ignore[private-member-access]
                scene, o, specular_reflection, diffuse_scattering
            ),
            max_order,
        )
        for o in order_list
    ]

    return (
        jnp.concatenate([c for c, _ in candidates_and_types], axis=0),
        jnp.concatenate([t for _, t in candidates_and_types], axis=0),
    )


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
        scene: "Scene",
        order: int | Sequence[int] | slice,
        specular_reflection: bool = True,
        diffuse_scattering: bool = False,
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        order = _normalize_order(order)
        return _generate_path_candidates_for_orders(
            self, scene, order, specular_reflection, diffuse_scattering
        )

    def _build_graph(
        self,
        scene: "Scene",
    ) -> tuple[CompleteGraph | DiGraph, int, int]:
        """Build the (optionally mask-filtered) graph used to enumerate path candidates.

        Shared by :meth:`_generate_path_candidates_for_one_order` and
        :meth:`generate_path_candidates_chunks_iter`, which only diverge on
        how they walk the resulting graph (:meth:`~differt_core.geometry.CompleteGraph.all_paths_array`
        vs. :meth:`~differt_core.geometry.CompleteGraph.all_paths_array_chunks`).

        Args:
            scene: The scene.

        Returns:
            A tuple of ``(graph, from_, to)``.
        """
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

        return graph, from_, to

    def _generate_path_candidates_for_one_order(
        self,
        scene: "Scene",
        order: int,
        specular_reflection: bool,  # ruff:ignore[unused-method-argument]
        diffuse_scattering: bool,  # ruff:ignore[unused-method-argument]
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        graph, from_, to = self._build_graph(scene)
        assume_quads = scene.mesh.assume_quads

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
        scene: "Scene",
        order: int | Sequence[int] | slice,
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
        fallback = self._single_chunk_fallback(
            scene, order, *args, chunk_size=effective_chunk_size, **kwargs
        )
        if fallback is not None:
            return fallback

        if not isinstance(order, int):
            msg = (
                "ExhaustivePathTracer.generate_path_candidates_chunks_iter does "
                "not support a sequence of orders; call "
                "generate_path_candidates(order=[...]) directly instead "
                "(without chunking)."
            )
            raise NotImplementedError(msg)

        (order,) = _normalize_order(order)

        graph, from_, to = self._build_graph(scene)
        assume_quads = scene.mesh.assume_quads

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
        scene: "Scene",
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
        scene: "Scene",
        order: int | Sequence[int] | slice,
        specular_reflection: bool = True,
        diffuse_scattering: bool = False,
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        order = _normalize_order(order)
        return _generate_path_candidates_for_orders(
            self, scene, order, specular_reflection, diffuse_scattering
        )

    def _build_visibility_graph(
        self,
        scene: "Scene",
    ) -> tuple[DiGraph, int, int]:
        """Build the visibility-pruned graph used to enumerate path candidates.

        Shared by :meth:`_generate_path_candidates_for_one_order` and
        :meth:`generate_path_candidates_chunks_iter`, which only diverge on
        how they walk the resulting graph (:meth:`~differt_core.geometry.CompleteGraph.all_paths_array`
        vs. :meth:`~differt_core.geometry.CompleteGraph.all_paths_array_chunks`).

        Args:
            scene: The scene.

        Returns:
            A tuple of ``(graph, from_, to)``.
        """
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

        return graph, from_, to

    def _generate_path_candidates_for_one_order(
        self,
        scene: "Scene",
        order: int,
        specular_reflection: bool,  # ruff:ignore[unused-method-argument]
        diffuse_scattering: bool,  # ruff:ignore[unused-method-argument]
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        graph, from_, to = self._build_visibility_graph(scene)
        assume_quads = scene.mesh.assume_quads

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
        scene: "Scene",
        order: int | Sequence[int] | slice,
        *args: Any,
        chunk_size: int | None = None,
        pad_chunks: bool = False,  # ruff:ignore[unused-method-argument]
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
        fallback = self._single_chunk_fallback(
            scene, order, *args, chunk_size=effective_chunk_size, **kwargs
        )
        if fallback is not None:
            return fallback

        if not isinstance(order, int):
            msg = (
                "HybridPathTracer.generate_path_candidates_chunks_iter does "
                "not support a sequence of orders; call "
                "generate_path_candidates(order=[...]) directly instead "
                "(without chunking)."
            )
            raise NotImplementedError(msg)

        (order,) = _normalize_order(order)

        graph, from_, to = self._build_visibility_graph(scene)
        assume_quads = scene.mesh.assume_quads

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
        scene: "Scene",
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
        scene: "Scene",
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


_SBR_TRACE_EPSILON = 1e-5
"""Offset applied to ray origins before each hit test, matching the value
used by :func:`Mesh.first_triangle_hit_by_ray<differt.geometry.Mesh.first_triangle_hit_by_ray>`."""


@no_type_check
@wp.kernel
def _sbr_trace_kernel(
    mesh_id: wp.uint64,
    normals: wp.array(dtype=wp.vec3),
    ray_origins: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    max_order: int,
    epsilon: float,
    assume_quads: wp.bool,
    output: wp.array(dtype=wp.int32),
) -> None:  # pragma: no cover
    # 'output' is a flat, row-major '[num_rays, max_order]' buffer (reshaped
    # on the JAX side): a 2D 'wp.array2d' output currently incurs much
    # higher overhead through the JAX/Warp FFI bridge than a flat one. It is
    # pre-filled with '-1' (inactive) by the caller; only genuine hits are
    # written below, before a ray goes inactive.
    tid = wp.tid()
    row = tid * max_order

    origin = ray_origins[tid]
    direction = ray_directions[tid]

    for i in range(max_order):
        query_origin = origin + direction * epsilon
        res = wp.mesh_query_ray(mesh_id, query_origin, direction, wp.inf)

        if not res.result:
            # Once a ray exits the scene, it (and all of its future
            # bounces) stays inactive: leave the rest of the row as '-1'.
            break

        face = res.face
        if assume_quads:
            # Store the primitive (quad) index, encoded as the index of the
            # first of its two triangles, matching the convention used by
            # 'ExhaustivePathTracer' and 'HybridPathTracer'.
            output[row + i] = (face // 2) * 2
        else:
            output[row + i] = face

        origin = origin + direction * (res.t + epsilon)
        normal = normals[face]
        direction = direction - 2.0 * wp.dot(direction, normal) * normal


@no_type_check
def _sbr_trace_func(
    mesh_id: int,
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_order: int,
    assume_quads: wp.bool,
    output: wp.array[wp.int32],
) -> None:
    wp_mesh = _get_warp_mesh(mesh_id, points, indices)
    output.fill_(-1)
    _warp_launch(
        _sbr_trace_kernel,
        dim=ray_origins.shape[0],
        inputs=[
            wp_mesh.id,
            normals,
            _Batched(ray_origins),
            _Batched(ray_directions),
            max_order,
            _SBR_TRACE_EPSILON,
            assume_quads,
        ],
        outputs=[_Batched(output, row_size=max_order)],
        device=ray_origins.device,
    )


def _sbr_trace(
    mesh: Mesh,
    flat_ray_origins: Float[Array, "num_rays_total 3"],
    flat_ray_directions: Float[Array, "num_rays_total 3"],
    max_order: int,
) -> Int[Array, "num_rays_total max_order"]:
    """Discover, for every ray, the sequence of primitives it bounces on.

    Unlike :meth:`Mesh.first_triangle_hit_by_ray<differt.geometry.Mesh.first_triangle_hit_by_ray>`,
    which performs a single hit test per call, this launches a single Warp
    kernel that performs the *entire* bounce loop for every ray, with a
    per-ray early exit as soon as a ray leaves the scene. Compared to
    calling :meth:`Mesh.first_triangle_hit_by_ray<differt.geometry.Mesh.first_triangle_hit_by_ray>`
    once per bounce from a ``jax.lax.scan`` (as a naive implementation
    would), this avoids ``max_order`` separate JAX/Warp round-trips and the
    corresponding intermediate ray-state materialization in between.

    Path candidates (triangle indices) are never differentiated (only the
    downstream, exact image-method solve is), so this helper is
    intentionally forward-only: gradients are stopped on all inputs.

    Args:
        mesh: The scene mesh.
        flat_ray_origins: The (flattened) ray origins.
        flat_ray_directions: The (flattened) ray directions.
        max_order: The maximum number of bounces to simulate.

    Returns:
        For each launched ray, the sequence of (at most ``max_order``)
        primitive indices it bounced on, using the same ``-1`` placeholder
        convention as path candidates.
    """
    triangles = mesh.triangles
    if mesh.mask is not None:
        triangles = jnp.where(mesh.mask[:, None], mesh.triangles, 0)

    mesh_id = np.uint64(id(mesh))
    num_rays = flat_ray_origins.shape[0]

    (recorded,) = wp.jax_callable(
        _sbr_trace_func,
        output_dims=(num_rays * max_order,),
        graph_mode=wp.JaxCallableGraphMode.NONE,
    )(
        mesh_id,
        jax.lax.stop_gradient(mesh.vertices),
        triangles.ravel(),
        jax.lax.stop_gradient(mesh.normals),
        jax.lax.stop_gradient(flat_ray_origins),
        jax.lax.stop_gradient(flat_ray_directions),
        max_order,
        mesh.assume_quads,
    )
    return jax.lax.stop_gradient(recorded).reshape(num_rays, max_order)


class SBRPathTracer(HybridPathTracer):
    """
    Shooting-and-bouncing ray (SBR) path tracer.

    Instead of enumerating a (possibly visibility-pruned) complete graph, like
    :class:`ExhaustivePathTracer` and :class:`HybridPathTracer` do, this tracer
    *discovers* candidate interaction sequences by launching a fixed, bounded
    population of rays from each transmitter and following their specular
    bounces, closely following the shooting-and-bouncing-rays (SBR) candidate
    generation procedure used by Sionna RT :cite:`sionna-rt`; see its
    `technical report <https://nvlabs.github.io/sionna/rt/tech-report/S3.html#SS1>`_
    for a detailed description of the algorithm this class is based on.

    Every ray trajectory yields (at most) one candidate sequence of primitive
    indices, so the memory needed to generate candidates only depends on
    :attr:`num_rays` and :attr:`max_num_candidates`, and no longer grows
    combinatorially with ``order`` or the number of primitives in the scene.
    Because many rays typically converge onto the same discrete sequence of
    primitives, especially at low orders, the discovered candidates are
    deduplicated (and bounded by :attr:`max_num_candidates`)
    before being passed to the same exact image-method solver used by
    :class:`ExhaustivePathTracer` and :class:`HybridPathTracer`.

    When ``order`` is a sequence (or a :class:`range`/``slice``), rays are
    launched only once, up to the maximum requested order: candidates for
    every requested order (including order 0 for line-of-sight if requested)
    are collected, padded up to the maximum requested order with ``-1``
    placeholders, and combined into a single array bounded by
    :attr:`max_num_candidates`.

    .. important::

        Because candidates are discovered from a finite ray population, this
        tracer is **not guaranteed to be exhaustive**: it may miss valid paths
        that subtend a small solid angle as seen from the transmitters,
        especially at high orders or in scenes with many small primitives.
        Increasing :attr:`num_rays` improves coverage, at the cost of memory
        and runtime.

    .. important::

        Like :class:`HybridPathTracer`, this tracer is best used for a small
        number of transmitters (rays are only launched from transmitters, not
        receivers).
    """

    max_num_candidates: int = int(1e5)
    """The maximum number of (deduplicated) path candidates that are kept.

    If more unique candidates are discovered than this value, the extra
    candidates are silently dropped.
    """

    def _launch_and_record(
        self,
        scene: "Scene",
        max_order: int,
    ) -> Int[Array, "num_rays_total max_order"]:
        """Launch rays and record the sequence of primitives each one hits.

        This is the core, discovery-only part of :meth:`generate_path_candidates`:
        it does not deduplicate nor bound the result, so that a single ray
        population can be shared to generate candidates for multiple orders
        at once (see :meth:`generate_path_candidates`, with a sequence of
        orders), by keeping only the trajectories whose natural number of
        interactions (i.e., where each one stops) matches one of the
        requested orders.

        Args:
            scene: The scene.
            max_order: The maximum number of bounces to simulate.

        Returns:
            For each launched ray, the sequence of (at most ``max_order``)
            primitive indices it bounced on, using the same ``-1`` placeholder
            convention as path candidates: once a ray exits the scene (or
            hits a masked-out triangle), all of its remaining, unrealized
            bounces are set to ``-1``.
        """
        mesh = scene.mesh

        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)
        num_tx_vertices = tx_vertices.shape[0]

        world_vertices = jnp.concatenate(
            (mesh.triangle_vertices.reshape(-1, 3), rx_vertices), axis=0
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

        # A single Warp kernel launch performs the entire bounce loop, for
        # every ray, with a per-ray early exit; see '_sbr_trace'.
        return _sbr_trace(
            mesh,
            ray_origins.reshape(-1, 3),
            ray_directions.reshape(-1, 3),
            max_order,
        )

    def _deduplicate_candidates(
        self,
        candidates: Int[Array, "num_rays_total order"],
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        """Deduplicate discovered ray trajectories into a bounded buffer.

        The output size is bounded by :attr:`max_num_candidates`, regardless
        of the order or the number of rays that discovered each sequence.

        Args:
            candidates: The (possibly duplicated, unbounded) discovered
                trajectories, see :meth:`_launch_and_record`.

        Returns:
            The deduplicated, bounded path candidates and interaction types.
        """
        path_candidates = jnp.unique(candidates, axis=0).astype(int)
        if path_candidates.shape[0] > self.max_num_candidates:
            path_candidates = path_candidates[: self.max_num_candidates]

        # Default: all specular reflections (value 0);
        # -1 marks inactive/padded interactions (e.g., lower-order candidates
        # padded with placeholders up to max_order).
        interaction_types = jnp.where(path_candidates >= 0, 0, -1).astype(jnp.int32)

        return path_candidates, interaction_types

    def generate_path_candidates(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
        specular_reflection: bool = True,  # ruff:ignore[unused-method-argument]
        diffuse_scattering: bool = False,  # ruff:ignore[unused-method-argument]
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        single_order = isinstance(order, int)
        order = _normalize_order(order)

        if single_order:
            (order,) = order

            if order == 0:
                path_candidates = jnp.zeros((1, 0), dtype=int)
                interaction_types = jnp.zeros((1, 0), dtype=jnp.int32)
                return path_candidates, interaction_types

            raw = self._launch_and_record(scene, order)
            hits = raw[:, :order]
            hits = hits[hits[:, order - 1] >= 0]

            return self._deduplicate_candidates(hits)

        order_list = sorted({int(o) for o in order})

        if not order_list:
            msg = "You must provide at least one order when 'order' is a sequence."
            raise ValueError(msg)

        max_order = order_list[-1]

        if max_order == 0:  # 'order_list' can only be '[0]' in this case.
            return self.generate_path_candidates(scene, 0)

        # A single ray population is shared across all requested orders,
        # launched once, up to 'max_order' bounces. For each requested order,
        # the prefix of each ray trajectory up to that depth (for rays that
        # completed at least that many bounces) is collected, padded with
        # placeholders up to 'max_order', and combined into a single array
        # bounded by 'max_num_candidates'.
        raw = self._launch_and_record(scene, max_order)
        all_candidates: list[Int[Array, "... max_order"]] = []
        for o in order_list:
            if o == 0:
                all_candidates.append(jnp.full((1, max_order), -1, dtype=int))
            else:
                hits = raw[:, :o]
                hits = hits[hits[:, o - 1] >= 0]
                hits = jnp.unique(hits, axis=0).astype(int)
                cands_o, _ = _pad_path_candidates(
                    hits, jnp.zeros_like(hits, dtype=jnp.int32), max_order
                )
                all_candidates.append(cands_o)

        combined = jnp.concatenate(all_candidates, axis=0)
        return self._deduplicate_candidates(combined)

    def generate_path_candidates_chunks_iter(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
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
        """Fall back to the default slice-based chunking.

        Unlike :class:`HybridPathTracer`, this tracer does not build a
        visibility graph, so there is nothing to chunk natively: candidates
        are generated all at once (bounded by :attr:`max_num_candidates`)
        and then sliced into chunks.

        Returns:
            An iterator over path candidates chunks.
        """
        effective_chunk_size = chunk_size or self.chunk_size
        fallback = self._single_chunk_fallback(
            scene, order, *args, chunk_size=effective_chunk_size, **kwargs
        )
        if fallback is not None:
            return fallback

        return AbstractPathTracer.generate_path_candidates_chunks_iter(
            self,
            scene,
            order,
            *args,
            chunk_size=effective_chunk_size,
            pad_chunks=pad_chunks,
            **kwargs,
        )


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


class _SBRPathTracerKwargs(TypedDict, total=False):
    num_rays: int
    epsilon: Float[ArrayLike, ""] | None
    hit_tol: Float[ArrayLike, ""] | None
    min_len: Float[ArrayLike, ""] | None
    smoothing_factor: Float[ArrayLike, ""] | None
    confidence_threshold: Float[ArrayLike, ""]
    batch_size: int | None
    chunk_size: int | None
    max_num_candidates: int


__all__ = [
    "_ExhaustivePathTracerKwargs",
    "_HybridPathTracerKwargs",
    "_SBRPathLauncherKwargs",
    "_SBRPathTracerKwargs",
]

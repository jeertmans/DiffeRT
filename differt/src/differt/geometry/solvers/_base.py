import abc
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.geometry._mesh import Mesh
from differt.geometry._paths import LaunchedPaths, TracedPaths
from differt.geometry._solver_image_method import (
    consecutive_vertices_are_on_same_side_of_mirror,
    image_method,
)
from differt.geometry._utils import (
    SizedIterator,
    assemble_path,
    check_path_candidates,
    ray_intersect_any_triangle,
    ray_intersect_triangle,
)
from differt.utils import smoothing_function

from ._dispatch import solve_mixed_interaction_paths

if TYPE_CHECKING:
    from differt.em import InteractionType
    from differt.geometry._scene import Scene

    from ._exhaustive import ExhaustivePathTracer
    from ._hybrid import HybridPathTracer


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
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> tuple[
        Int[Array, "... num_path_candidates max_order"],
        Int[Array, "... num_path_candidates max_order"],
    ]:
        """
        Return a tuple of ``(path_candidates, interaction_types)``.

        ``path_candidates`` contains primitive indices: a (quad-aware)
        triangle index for ``REFLECTION``, ``SCATTERING``, and
        ``TRANSMISSION`` bounces, or a flat half-edge index (``3 *
        triangle_index + local_edge_index``) for ``DIFFRACTION`` bounces,
        matching :attr:`Mesh.wedge_angles<differt.geometry.Mesh.wedge_angles>`.
        ``interaction_types`` classifies the bounce, matching
        :class:`InteractionType<differt.em.InteractionType>`. A value of
        ``-1`` in either array indicates an "inactive" interaction or
        padded bounce.

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
            allowed_interactions: The set of interaction types a bounce may
                take. Defaults to
                ``frozenset({InteractionType.REFLECTION})`` (today's
                behavior) when :data:`None`. :class:`SBRPathTracer` only
                supports ``REFLECTION`` for now (its ray-shooting kernel
                does not (yet) continue through diffraction edges or
                transmissive faces).

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
    ) -> (
        SizedIterator[
            tuple[
                Int[Array, "... chunk_size max_order"],
                Int[Array, "... chunk_size max_order"],
            ]
        ]
        | Iterator[
            tuple[
                Int[Array, "... chunk_size max_order"],
                Int[Array, "... chunk_size max_order"],
            ]
        ]
    ):
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
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> TracedPaths:
        """Core logic to trace the exact paths from the proposed candidates.

        Args:
            scene: The scene.
            path_candidates: Primitive indices for each candidate, see
                :meth:`generate_path_candidates`.
            interaction_types: Interaction type for each bounce.
            allowed_interactions: The set of interaction types that was used
                to generate ``path_candidates`` (see
                :meth:`generate_path_candidates`): needed again here because
                it determines *how* mixed-type candidates are geometrically
                solved (e.g., whether a DIFFRACTION bounce may be present,
                requiring the general Fermat solver instead of the exact
                image method), which must be resolved ahead of any traced
                computation. Defaults to
                ``frozenset({InteractionType.REFLECTION})`` when :data:`None`.

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
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> TracedPaths: ...

    @overload
    def trace_paths(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
        chunk_size: int,
        pad_chunks: bool = False,
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> SizedIterator[TracedPaths] | Iterator[TracedPaths]: ...

    def trace_paths(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
        chunk_size: int | None = None,
        pad_chunks: bool = False,
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> TracedPaths | SizedIterator[TracedPaths] | Iterator[TracedPaths]:
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
            allowed_interactions: The set of interaction types a bounce may
                take, see :meth:`generate_path_candidates`.

        Returns:
            Traced paths, or a sized iterator thereof. If ``chunk_size`` is
            set and :meth:`generate_path_candidates_chunks_iter` was
            overridden to return a plain (unsized) iterator, a plain
            iterator is returned instead of a
            :class:`~differt.geometry.SizedIterator`.

        Raises:
            NotImplementedError: If ``order`` is a sequence of orders and
                ``chunk_size`` is not :data:`None`.
        """
        if not isinstance(order, int) and chunk_size is not None:
            msg = "Chunked generation ('chunk_size') is not supported when 'order' is a sequence of orders."  # TODO: implement me, this should be relatively easy to do
            raise NotImplementedError(msg)

        if chunk_size is not None:
            chunks_iter = self.generate_path_candidates_chunks_iter(
                scene,
                order,
                allowed_interactions,
                chunk_size=chunk_size,
                pad_chunks=pad_chunks,
            )
            traced_chunks = (
                self.trace_path_candidates(scene, cands, types, allowed_interactions)
                for cands, types in chunks_iter
            )
            if hasattr(chunks_iter, "__len__"):
                return SizedIterator(traced_chunks, size=chunks_iter.__len__)
            # Custom overrides of 'generate_path_candidates_chunks_iter' are
            # allowed to return a plain (unsized) iterator, see its docstring;
            # fall back to a plain iterator too, instead of faking a size.
            return traced_chunks
        candidates, interactions = self.generate_path_candidates(
            scene, order, allowed_interactions
        )
        return self.trace_path_candidates(
            scene, candidates, interactions, allowed_interactions
        )


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

    def bounce_rays(  # ruff: ignore[no-self-use]
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
        scene: "Scene",  # ruff: ignore[unused-method-argument]
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
    use_fermat: bool = False,
    needs_splice: bool = False,
) -> TracedPaths:
    """
    Trace exact paths from the proposed candidates.

    ``use_fermat`` and ``needs_splice`` are Python-level (non-traced) flags,
    known ahead of time from the caller's ``allowed_interactions`` (whether
    it contains DIFFRACTION and/or TRANSMISSION, respectively): unlike
    ``interaction_types``'s actual values, they must be resolved outside of
    any traced computation, since they select which (mutually exclusive)
    geometric solver is used, see
    :func:`~differt.geometry.solvers._dispatch.solve_mixed_interaction_paths`.
    When both are :data:`False` (the default, reflection/scattering-only
    case), this reduces exactly to the original single
    :func:`~differt.geometry.image_method` call.

    Returns:
        The traced paths.
    """
    from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

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

    # [num_path_candidates order] - normalize a missing 'interaction_types'
    # (legacy direct-call convention) to all-REFLECTION for active bounces.
    if interaction_types is None:
        interaction_types_norm = jnp.where(
            active, InteractionType.REFLECTION, InteractionType.NONE
        )
    else:
        interaction_types_norm = interaction_types

    # [num_path_candidates order] - bounces whose validity checks are
    # edge- rather than triangle-based (DIFFRACTION), and bounces that do
    # not bend the ray at all (TRANSMISSION); both are all-'False' in the
    # default (reflection/scattering-only) case, in which case every check
    # below reduces exactly to today's behavior.
    is_diffraction = active & (interaction_types_norm == InteractionType.DIFFRACTION)
    is_transmission = active & (interaction_types_norm == InteractionType.TRANSMISSION)

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
    elif use_fermat or needs_splice:
        # Mixed (non-reflection-only) interaction types: reflection/
        # scattering/diffraction bounces are solved jointly (they bend the
        # ray and are not independent of one another); transmission
        # bounces do not bend the ray, and are spliced in afterward. See
        # 'solve_mixed_interaction_paths' for the full algorithm.
        full_paths = solve_mixed_interaction_paths(
            mesh,
            tx_vertices,
            rx_vertices,
            path_candidates,
            interaction_types_norm,
            use_fermat=use_fermat,
            needs_splice=needs_splice,
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
    # (DIFFRACTION bounces do not land on a triangle at all -- 'active' is
    # narrowed to exclude them here, and they get their own edge-segment
    # check below instead, folded into 'inside_triangles').

    # [num_path_candidates order]
    active_for_triangle_check = active & ~is_diffraction

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
            inside_triangles = jnp.where(
                active_for_triangle_check, inside_triangles, 1.0
            ).min(
                axis=-1, initial=1.0
            )  # Reduce on 'order' axis, ignoring placeholder/diffraction interactions
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
            inside_triangles = (inside_triangles | ~active_for_triangle_check).all(
                axis=-1
            )  # Reduce on 'order' axis, ignoring placeholder/diffraction interactions
    elif smoothing_factor is not None:
        inside_triangles = ray_intersect_triangle(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
            smoothing_factor=smoothing_factor,
        )[1]
        inside_triangles = jnp.where(
            active_for_triangle_check, inside_triangles, 1.0
        ).min(
            axis=-1, initial=1.0
        )  # Reduce on 'order' axis, ignoring placeholder/diffraction interactions
    else:
        inside_triangles = ray_intersect_triangle(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            triangle_vertices,
            epsilon=epsilon,
        )[1]
        inside_triangles = (inside_triangles | ~active_for_triangle_check).all(
            axis=-1
        )  # Reduce on 'order' axis, ignoring placeholder/diffraction interactions

    if use_fermat:
        # 3.1b - Check if DIFFRACTION bounces lie within their finite edge segment
        prim0 = safe_path_candidates // 3
        local_edge = safe_path_candidates % 3
        edge_0 = mesh.triangle_edges[prim0, local_edge, 0, :]
        edge_1 = mesh.triangle_edges[prim0, local_edge, 1, :]
        edge_vector = edge_1 - edge_0
        # Note: this is *not* the same 'e_hat' as
        # 'Mesh._wedge_static_geometry' (used for the Fermat solve above),
        # whose sign follows its own UTD-specific convention and is not
        # guaranteed to point from 'edge_0' towards 'edge_1'; the Fermat
        # solve does not care (a line's affine span is sign-independent),
        # but this segment-membership check needs a direction consistently
        # oriented from 'edge_0' to 'edge_1'.
        edge_len = jnp.linalg.norm(edge_vector, axis=-1)
        edge_dir = edge_vector / jnp.where(
            edge_len[..., None] > 0, edge_len[..., None], 1.0
        )

        # [num_tx_vertices num_rx_vertices num_path_candidates order]
        bounce_vertices = full_paths[..., 1:-1, :]
        t = jnp.sum((bounce_vertices - edge_0) * edge_dir, axis=-1)
        violation = jnp.maximum(-t, t - edge_len)  # > 0 iff outside the segment

        active_for_edge_check = active & is_diffraction
        if smoothing_factor is not None:
            on_edge_segment = 1.0 - smoothing_function(violation, smoothing_factor)
            on_edge_segment = jnp.where(
                active_for_edge_check, on_edge_segment, 1.0
            ).min(axis=-1, initial=1.0)
            inside_triangles = inside_triangles * on_edge_segment
        else:
            on_edge_segment = violation <= 0.0
            on_edge_segment = (on_edge_segment | ~active_for_edge_check).all(axis=-1)
            inside_triangles = inside_triangles & on_edge_segment

    # 3.2 - Check if consecutive path vertices are on the same side of mirrors
    # (does not apply to DIFFRACTION, which is not a mirror reflection, nor
    # to TRANSMISSION, which passes through by design).

    # [num_path_candidates order]
    active_for_mirror_check = active & ~is_diffraction & ~is_transmission

    # [num_tx_vertices num_rx_vertices num_path_candidates]
    if smoothing_factor is not None:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirror(
            full_paths,
            mirror_vertices,
            mirror_normals,
            smoothing_factor=smoothing_factor,
        )
        valid_reflections = jnp.where(
            active_for_mirror_check, valid_reflections, 1.0
        ).min(
            axis=-1, initial=1.0
        )  # Reduce on 'order', ignoring placeholder/diffraction/transmission interactions
    else:
        valid_reflections = consecutive_vertices_are_on_same_side_of_mirror(
            full_paths,
            mirror_vertices,
            mirror_normals,
        )
        valid_reflections = (valid_reflections | ~active_for_mirror_check).all(
            axis=-1
        )  # Reduce on 'order', ignoring placeholder/diffraction/transmission interactions

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
    allowed_interactions: "frozenset[InteractionType]",
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
        allowed_interactions: The set of interaction types a bounce may take.

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
            *solver._generate_path_candidates_for_one_order(  # ruff: ignore[private-member-access]
                scene, o, allowed_interactions
            ),
            max_order,
        )
        for o in order_list
    ]

    return (
        jnp.concatenate([c for c, _ in candidates_and_types], axis=0),
        jnp.concatenate([t for _, t in candidates_and_types], axis=0),
    )

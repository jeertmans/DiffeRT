from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

__all__ = ["HybridPathTracer", "_HybridPathTracerKwargs"]

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int

from differt.geometry._interaction_sites import (
    InteractionSites,
    build_interaction_sites,
    interaction_sites_mesh_mask,
    interaction_sites_valid_mask,
)
from differt.geometry._paths import TracedPaths
from differt.geometry._utils import SizedIterator
from differt_core.geometry import CompleteGraph, DiGraph

from ._base import (
    AbstractPathTracer,
    _generate_path_candidates_for_orders,
    _normalize_order,
    _trace_path_candidates,
)

if TYPE_CHECKING:
    from differt.em import InteractionType
    from differt.geometry._scene import Scene


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
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

        if allowed_interactions is None:
            allowed_interactions = frozenset({InteractionType.REFLECTION})

        order = _normalize_order(order)
        return _generate_path_candidates_for_orders(
            self, scene, order, allowed_interactions
        )

    def _build_visibility_graph(
        self,
        scene: "Scene",
        allowed_interactions: "frozenset[InteractionType]",
    ) -> tuple[DiGraph, int, int, InteractionSites]:
        """Build the visibility-pruned graph used to enumerate path candidates.

        Shared by :meth:`_generate_path_candidates_for_one_order` and
        :meth:`generate_path_candidates_chunks_iter`, which only diverge on
        how they walk the resulting graph (:meth:`~differt_core.geometry.CompleteGraph.all_paths_array`
        vs. :meth:`~differt_core.geometry.CompleteGraph.all_paths_array_chunks`).

        A DIFFRACTION site (half-edge) is considered visible from a vertex
        iff either of its two adjacent triangles is (a conservative
        over-approximation: it never prunes away a half-edge that might
        actually be reachable).

        Args:
            scene: The scene.
            allowed_interactions: The set of interaction types a bounce may take.

        Returns:
            A tuple of ``(graph, from_, to, sites)``.
        """
        from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

        tx_vertices = scene.transmitters.reshape(-1, 3)
        rx_vertices = scene.receivers.reshape(-1, 3)
        mesh = scene.mesh

        sites = build_interaction_sites(mesh, allowed_interactions)
        graph = CompleteGraph(sites.kind.shape[0])

        triangles_visible_from_tx = mesh.triangles_visible_from_vertex(
            tx_vertices,
            num_rays=self.num_rays,
        ).any(axis=0)
        triangles_visible_from_rx = mesh.triangles_visible_from_vertex(
            rx_vertices,
            num_rays=self.num_rays,
        ).any(axis=0)

        is_diffraction = sites.kind == InteractionType.DIFFRACTION
        if mesh.assume_quads:
            quad_visible_from_tx = triangles_visible_from_tx.reshape(-1, 2).any(axis=-1)
            quad_visible_from_rx = triangles_visible_from_rx.reshape(-1, 2).any(axis=-1)
        else:
            quad_visible_from_tx = triangles_visible_from_tx
            quad_visible_from_rx = triangles_visible_from_rx

        if InteractionType.DIFFRACTION in allowed_interactions:
            _, _, _, primn = mesh._wedge_static_geometry()  # ruff: ignore[private-member-access]
            primn = primn.ravel()
            prim0 = jnp.arange(mesh.num_triangles).repeat(3)
            edge_visible_from_tx = (
                triangles_visible_from_tx[prim0] | triangles_visible_from_tx[primn]
            )
            edge_visible_from_rx = (
                triangles_visible_from_rx[prim0] | triangles_visible_from_rx[primn]
            )
            safe_primitive = jnp.where(is_diffraction, sites.primitive, 0)
            visible_from_tx = jnp.where(
                is_diffraction,
                edge_visible_from_tx[safe_primitive],
                quad_visible_from_tx[jnp.where(is_diffraction, 0, sites.primitive)],
            )
            visible_from_rx = jnp.where(
                is_diffraction,
                edge_visible_from_rx[safe_primitive],
                quad_visible_from_rx[jnp.where(is_diffraction, 0, sites.primitive)],
            )
        else:
            visible_from_tx = quad_visible_from_tx
            visible_from_rx = quad_visible_from_rx

        graph = DiGraph.from_complete_graph(graph)
        from_, to = graph.insert_from_and_to_nodes(
            from_adjacency=np.asarray(visible_from_tx),
            to_adjacency=np.asarray(visible_from_rx),
        )

        # Only touch 'Mesh.diffraction_edges_mask' (which triggers a vertex
        # dedup + non-manifold-edge check) when DIFFRACTION is actually
        # allowed -- not needed otherwise, exactly as before.
        needs_diffraction_filter = InteractionType.DIFFRACTION in allowed_interactions
        needs_mask_filter = mesh.mask is not None

        if needs_diffraction_filter or needs_mask_filter:
            valid = (
                interaction_sites_valid_mask(mesh, sites)
                if needs_diffraction_filter
                else jnp.ones(sites.kind.shape[0], dtype=bool)
            )
            if needs_mask_filter:
                valid = valid & interaction_sites_mesh_mask(mesh, sites)
            graph.filter_by_mask(np.asarray(valid), fast_mode=True)

        return graph, from_, to, sites

    def _generate_path_candidates_for_one_order(
        self,
        scene: "Scene",
        order: int,
        allowed_interactions: "frozenset[InteractionType]",
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        graph, from_, to, sites = self._build_visibility_graph(
            scene, allowed_interactions
        )

        site_candidates = jnp.asarray(
            graph.all_paths_array(
                from_=from_,
                to=to,
                depth=order + 2,
                include_from_and_to=False,
            ),
            dtype=int,
        )

        path_candidates = sites.primitive[site_candidates]
        interaction_types = sites.kind[site_candidates]

        return path_candidates, interaction_types

    def generate_path_candidates_chunks_iter(
        self,
        scene: "Scene",
        order: int | Sequence[int] | slice,
        allowed_interactions: "frozenset[InteractionType] | None" = None,
        *args: Any,
        chunk_size: int | None = None,
        pad_chunks: bool = False,  # ruff: ignore[unused-method-argument]
        **kwargs: Any,
    ) -> (
        SizedIterator[
            tuple[
                Int[Array, "... chunk_size order"],
                Int[Array, "... chunk_size order"],
            ]
        ]
        | Iterator[
            tuple[
                Int[Array, "... chunk_size order"],
                Int[Array, "... chunk_size order"],
            ]
        ]
    ):
        """Override to support native chunked generation from the graph.

        Returns:
            An iterator over path candidates chunks.
        """
        from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

        if allowed_interactions is None:
            allowed_interactions = frozenset({InteractionType.REFLECTION})

        effective_chunk_size = chunk_size or self.chunk_size
        fallback = self._single_chunk_fallback(
            scene,
            order,
            allowed_interactions,
            *args,
            chunk_size=effective_chunk_size,
            **kwargs,
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

        graph, from_, to, sites = self._build_visibility_graph(
            scene, allowed_interactions
        )

        site_candidates_iter = graph.all_paths_array_chunks(
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
            for chunk_arr in site_candidates_iter:
                site_candidates_chunk = jnp.asarray(chunk_arr, dtype=int)
                candidates_chunk = sites.primitive[site_candidates_chunk]
                interaction_types_chunk = sites.kind[site_candidates_chunk]
                yield candidates_chunk, interaction_types_chunk

        if hasattr(site_candidates_iter, "__len__"):
            return SizedIterator(gen(), size=site_candidates_iter.__len__)

        return gen()

    @eqx.filter_jit
    def trace_path_candidates(
        self,
        scene: "Scene",
        path_candidates: Int[Array, "num_candidates order"],
        interaction_types: Int[Array, "num_candidates order"],
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> TracedPaths:
        from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

        if allowed_interactions is None:
            allowed_interactions = frozenset({InteractionType.REFLECTION})

        use_fermat = InteractionType.DIFFRACTION in allowed_interactions
        needs_splice = InteractionType.TRANSMISSION in allowed_interactions

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
            use_fermat=use_fermat,
            needs_splice=needs_splice,
        )


class _HybridPathTracerKwargs(TypedDict, total=False):
    num_rays: int
    epsilon: Float[ArrayLike, ""] | None
    hit_tol: Float[ArrayLike, ""] | None
    min_len: Float[ArrayLike, ""] | None
    smoothing_factor: Float[ArrayLike, ""] | None
    confidence_threshold: Float[ArrayLike, ""]
    batch_size: int | None
    chunk_size: int | None

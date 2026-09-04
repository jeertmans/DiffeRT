from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

__all__ = ["ExhaustivePathTracer", "_ExhaustivePathTracerKwargs"]

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

    def _build_graph(
        self,
        scene: "Scene",
        allowed_interactions: "frozenset[InteractionType]",
    ) -> tuple[CompleteGraph | DiGraph, int, int, InteractionSites]:
        """Build the (optionally mask-filtered) graph used to enumerate path candidates.

        Shared by :meth:`_generate_path_candidates_for_one_order` and
        :meth:`generate_path_candidates_chunks_iter`, which only diverge on
        how they walk the resulting graph (:meth:`~differt_core.geometry.CompleteGraph.all_paths_array`
        vs. :meth:`~differt_core.geometry.CompleteGraph.all_paths_array_chunks`).

        Args:
            scene: The scene.
            allowed_interactions: The set of interaction types a bounce may take.

        Returns:
            A tuple of ``(graph, from_, to, sites)``.
        """
        from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

        sites = build_interaction_sites(scene.mesh, allowed_interactions)
        graph = CompleteGraph(sites.kind.shape[0])

        # Structurally-invalid half-edge slots (not actual diffraction
        # edges) must always be excluded when DIFFRACTION is allowed;
        # mesh-mask filtering of the other sites remains opt-in, exactly
        # as it was for plain triangle indices.
        needs_diffraction_filter = InteractionType.DIFFRACTION in allowed_interactions
        needs_mask_filter = (
            self.disconnect_inactive_triangles and scene.mesh.mask is not None
        )

        if needs_diffraction_filter or needs_mask_filter:
            valid = (
                interaction_sites_valid_mask(scene.mesh, sites)
                if needs_diffraction_filter
                else jnp.ones(sites.kind.shape[0], dtype=bool)
            )
            if needs_mask_filter:
                valid = valid & interaction_sites_mesh_mask(scene.mesh, sites)

            graph = DiGraph.from_complete_graph(graph)
            from_, to = graph.insert_from_and_to_nodes()
            graph.filter_by_mask(np.asarray(valid), fast_mode=True)
        else:
            from_ = graph.num_nodes
            to = from_ + 1

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
        graph, from_, to, sites = self._build_graph(scene, allowed_interactions)

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
        pad_chunks: bool = False,
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

        # Use instance chunk_size if not explicitly provided
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
                "ExhaustivePathTracer.generate_path_candidates_chunks_iter does "
                "not support a sequence of orders; call "
                "generate_path_candidates(order=[...]) directly instead "
                "(without chunking)."
            )
            raise NotImplementedError(msg)

        (order,) = _normalize_order(order)

        graph, from_, to, sites = self._build_graph(scene, allowed_interactions)

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
                if pad_chunks and len(chunk_arr) < effective_chunk_size:
                    pad_width = ((0, effective_chunk_size - len(chunk_arr)), (0, 0))
                    padded_chunk = np.pad(
                        chunk_arr, pad_width, mode="constant", constant_values=-1
                    )
                else:
                    padded_chunk = chunk_arr

                site_candidates_chunk = jnp.asarray(padded_chunk, dtype=int)
                # '-1' (padding) must stay '-1' after the site->primitive/kind
                # lookup -- a raw negative index would otherwise silently
                # wrap around to the *last* site instead.
                chunk_active = site_candidates_chunk >= 0
                safe_site_candidates_chunk = jnp.where(
                    chunk_active, site_candidates_chunk, 0
                )
                candidates_chunk = jnp.where(
                    chunk_active, sites.primitive[safe_site_candidates_chunk], -1
                )
                interaction_types_chunk = jnp.where(
                    chunk_active,
                    sites.kind[safe_site_candidates_chunk],
                    InteractionType.NONE,
                )
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


class _ExhaustivePathTracerKwargs(TypedDict, total=False):
    epsilon: Float[ArrayLike, ""] | None
    hit_tol: Float[ArrayLike, ""] | None
    min_len: Float[ArrayLike, ""] | None
    smoothing_factor: Float[ArrayLike, ""] | None
    confidence_threshold: Float[ArrayLike, ""]
    batch_size: int | None
    disconnect_inactive_triangles: bool
    chunk_size: int | None

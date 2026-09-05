from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict, no_type_check

__all__ = [
    "SBRPathLauncher",
    "SBRPathTracer",
    "_SBRPathLauncherKwargs",
    "_SBRPathTracerKwargs",
]

import jax
import jax.numpy as jnp
import warp as wp
from jaxtyping import Array, ArrayLike, Float, Int

from differt.geometry._mesh import Mesh
from differt.geometry._utils import SizedIterator, fibonacci_lattice, viewing_frustum
from differt.geometry._warp_utils import _Batched, _warp_launch

from ._base import (
    AbstractPathLauncher,
    AbstractPathTracer,
    _normalize_order,
    _pad_path_candidates,
)
from ._hybrid import HybridPathTracer

if TYPE_CHECKING:
    from differt.em import InteractionType
    from differt.geometry._scene import Scene


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
        normal = res.normal
        direction = direction - 2.0 * wp.dot(direction, normal) * normal


@no_type_check
def _sbr_trace_func(
    mesh_points: wp.array[wp.vec3],
    mesh_indices: wp.array[wp.int32],
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_order: int,
    assume_quads: wp.bool,
    output: wp.array[wp.int32],
) -> None:
    wp_mesh = wp.Mesh(points=mesh_points, indices=mesh_indices)
    output.fill_(-1)
    _warp_launch(
        _sbr_trace_kernel,
        dim=ray_origins.shape[0],
        inputs=[
            wp_mesh.id,
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

    num_rays = flat_ray_origins.shape[0]

    (recorded,) = wp.jax_callable(
        _sbr_trace_func,
        output_dims=(num_rays * max_order,),
        graph_mode=wp.JaxCallableGraphMode.NONE,
    )(
        jax.lax.stop_gradient(mesh.vertices),
        jax.lax.stop_gradient(triangles.ravel()),
        jax.lax.stop_gradient(flat_ray_origins),
        jax.lax.stop_gradient(flat_ray_directions),
        max_order,
        mesh.assume_quads,
    )
    return recorded.reshape(num_rays, max_order)


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
        allowed_interactions: "frozenset[InteractionType] | None" = None,
    ) -> tuple[
        Int[Array, "num_candidates order"],
        Int[Array, "num_candidates order"],
    ]:
        from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

        if allowed_interactions is None:
            allowed_interactions = frozenset({InteractionType.REFLECTION})
        if not allowed_interactions <= {InteractionType.REFLECTION}:
            msg = (
                f"{type(self).__name__} only supports 'REFLECTION' for "
                "'allowed_interactions': its ray-shooting Warp kernel does not "
                "(yet) continue through diffraction edges or transmissive "
                "faces. Use 'ExhaustivePathTracer' or 'HybridPathTracer' for "
                "non-reflection interactions."
            )
            raise NotImplementedError(msg)

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

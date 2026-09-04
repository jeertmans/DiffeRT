"""Mixed-interaction-type geometric path solving (reflection/scattering/diffraction/transmission)."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from differt.geometry._mesh import Mesh
from differt.geometry._solver_fermat import fermat_path_on_linear_objects
from differt.geometry._solver_image_method import (
    image_method,
    intersection_of_ray_with_plane,
)
from differt.geometry._utils import assemble_path, orthogonal_basis


def _bending_first_permutation(
    is_bending: Bool[Array, "*batch order"],
) -> Int[Array, "*batch order"]:
    """
    Compute, per row, a permutation that stably moves bending entries to the front.

    "Bending" (``True``) entries of the last axis are moved to the front, in
    their original relative order, followed by all other ("non-bending",
    ``False``) entries, also in their original relative order. This turns an
    arbitrary interleaving of bending (reflection/scattering/diffraction) and
    non-bending (transmission, padding) bounces into a layout where the
    non-bending bounces form a trailing suffix, exactly like today's
    trailing ``-1`` placeholder convention -- so the very same
    receiver-collapse trick used for padding can be reused for them.

    Args:
        is_bending: Whether each bounce bends the ray.

    Returns:
        A permutation of ``0..order-1`` for each row, suitable for
        ``jnp.take_along_axis(x, perm, axis=-1)``.
    """
    # Stable sort key: 0 for bending (goes first), 1 for non-bending.
    key = (~is_bending).astype(jnp.int32)
    return jnp.argsort(key, axis=-1, stable=True)


def _nearest_real_neighbor_indices(
    is_real: Bool[Array, "*batch length"],
) -> tuple[Int[Array, "*batch length"], Int[Array, "*batch length"]]:
    """
    Find, for each position, the nearest "real" position at or before/after it.

    Used to find, for a non-bending (transmission) bounce, the two nearest
    surrounding "real" vertices (the transmitter/receiver, or a solved
    reflection/scattering/diffraction bounce) that the straight segment
    passing through it connects.

    Args:
        is_real: Whether each position along the last axis is "real".
            ``is_real[..., 0]`` and ``is_real[..., -1]`` must be ``True``
            (e.g., path endpoints), so both outputs are always well defined.

    Returns:
        A tuple ``(prev_idx, next_idx)``, the index of the nearest ``True``
        position at or before, and at or after, each position.
    """
    length = is_real.shape[-1]
    idx = jnp.broadcast_to(jnp.arange(length), is_real.shape)

    marked_fwd = jnp.where(is_real, idx, -1)
    prev_idx = jax.lax.associative_scan(jnp.maximum, marked_fwd, axis=-1)

    marked_bwd = jnp.where(is_real, idx, length)[..., ::-1]
    next_idx = jax.lax.associative_scan(jnp.minimum, marked_bwd, axis=-1)[..., ::-1]

    return prev_idx, next_idx


def _surface_geometry(
    mesh: Mesh, safe_path_candidates: Int[Array, "num_path_candidates order"]
) -> tuple[
    Float[Array, "num_path_candidates order 3"],
    Float[Array, "num_path_candidates order 3"],
]:
    """Surface (mirror/transmissive-face) vertex and normal for REFLECTION/SCATTERING/TRANSMISSION slots.

    ``safe_path_candidates`` is clipped to a valid triangle/quad index
    first, so this stays safe to call on slots that hold a half-edge index
    instead (DIFFRACTION), or a placeholder; callers are expected to
    discard those entries, exactly as
    :func:`~differt.em.GeometricFieldSolver._surface_interaction_geometry`
    already does for the EM solver side.

    Returns:
        A tuple ``(surface_vertex, surface_normal)``.
    """
    num_triangles = mesh.triangles.shape[0]
    safe_surface = jnp.clip(safe_path_candidates, 0, num_triangles - 1)
    surface_vertex = mesh.triangle_vertices[safe_surface, 0, :]
    surface_normal = mesh.normals[safe_surface]
    return surface_vertex, surface_normal


def _edge_geometry(
    mesh: Mesh, safe_path_candidates: Int[Array, "num_path_candidates order"]
) -> tuple[
    Float[Array, "num_path_candidates order 3"],
    Float[Array, "num_path_candidates order 3"],
]:
    """Edge origin and (unit) direction for DIFFRACTION slots.

    ``safe_path_candidates`` is interpreted as a flat half-edge index
    (``3 * triangle_index + local_edge_index``); it is always in-bounds
    for :meth:`Mesh._wedge_static_geometry` regardless of the actual
    interaction type of the slot, so this is safe to call unconditionally
    too (see :func:`_surface_geometry`).

    Returns:
        A tuple ``(edge_origin, edge_direction)``.
    """
    prim0 = safe_path_candidates // 3
    local_edge = safe_path_candidates % 3
    e0 = mesh.triangle_edges[prim0, local_edge, 0, :]
    _, _, e_hat_table, _ = mesh._wedge_static_geometry()  # ruff: ignore[private-member-access]
    e_hat = e_hat_table[prim0, local_edge, :]
    return e0, e_hat


def _bending_geometry_for_fermat(
    mesh: Mesh,
    safe_path_candidates: Int[Array, "num_path_candidates order"],
    is_diffraction: Bool[Array, "num_path_candidates order"],
) -> tuple[
    Float[Array, "num_path_candidates order 3"],
    Float[Array, "num_path_candidates order 2 3"],
]:
    """Build ``(object_origins, object_vectors)`` mixing surface (2-D) and edge (1-D) objects.

    Matches :func:`~differt.geometry.fermat_path_on_linear_objects`'s
    convention: a plane is described by its vertex plus 2 orthogonal
    in-plane vectors; an edge, by one of its endpoints plus its direction,
    with the second vector row set to zero.

    Returns:
        A tuple ``(object_origins, object_vectors)``.
    """
    surface_vertex, surface_normal = _surface_geometry(mesh, safe_path_candidates)
    e0, e_hat = _edge_geometry(mesh, safe_path_candidates)

    object_origin = jnp.where(is_diffraction[..., None], e0, surface_vertex)

    v, w = orthogonal_basis(surface_normal)
    plane_vectors = jnp.stack((v, w), axis=-2)
    edge_vectors = jnp.stack((e_hat, jnp.zeros_like(e_hat)), axis=-2)
    object_vectors = jnp.where(
        is_diffraction[..., None, None], edge_vectors, plane_vectors
    )

    return object_origin, object_vectors


def solve_mixed_interaction_paths(
    mesh: Mesh,
    tx_vertices: Float[Array, "num_tx_vertices 3"],
    rx_vertices: Float[Array, "num_rx_vertices 3"],
    path_candidates: Int[Array, "num_path_candidates order"],
    interaction_types: Int[Array, "num_path_candidates order"],
    *,
    use_fermat: bool,
    needs_splice: bool,
    fermat_kwargs: dict[str, Any] | None = None,
) -> Float[Array, "num_tx_vertices num_rx_vertices num_path_candidates order+2 3"]:
    """
    Solve intermediate path vertices for a batch of (possibly mixed-type) path candidates.

    ``REFLECTION``, ``SCATTERING``, and ``DIFFRACTION`` bounces are "bending"
    interactions: their positions are found jointly (they are not
    independent of one another), via
    :func:`~differt.geometry.image_method` when only
    ``REFLECTION``/``SCATTERING`` are involved (``use_fermat=False``), or
    via :func:`~differt.geometry.fermat_path_on_linear_objects` otherwise
    (mixing 2-D reflection/scattering planes and 1-D diffraction edges in a
    single call). ``TRANSMISSION`` does not bend the ray (current slab
    model): when ``needs_splice=True``, bending positions are first solved
    as if transmission bounces were not part of the path at all (by
    stably reordering each row so that bending bounces come first, reusing
    the existing trailing-placeholder receiver-collapse trick for the
    rest), and each transmission bounce's position is then computed
    afterward, as the intersection of the straight segment between its two
    nearest surrounding "real" vertices (solved bends, or the transmitter/
    receiver) with its transmissive face.

    Args:
        mesh: The scene mesh.
        tx_vertices: The transmitter vertices.
        rx_vertices: The receiver vertices.
        path_candidates: Primitive indices for each candidate: a
            (quad-aware) triangle index for REFLECTION/SCATTERING/
            TRANSMISSION, a flat half-edge index for DIFFRACTION, or ``-1``
            for a placeholder/padded bounce.
        interaction_types: The interaction type of each bounce, matching
            :class:`~differt.em.InteractionType`, with ``-1`` for a
            placeholder/padded bounce.
        use_fermat: Whether any candidate may contain a DIFFRACTION bounce,
            requiring the general Fermat solver for the bending sub-path
            (a Python-level, non-traced flag: known ahead of time from the
            caller's ``allowed_interactions``).
        needs_splice: Whether any candidate may contain a TRANSMISSION
            bounce, requiring the reorder-and-splice procedure described
            above (also a Python-level, non-traced flag).
        fermat_kwargs: Extra keyword arguments forwarded to
            :func:`~differt.geometry.fermat_path_on_linear_objects` (only
            used when ``use_fermat`` is set).

    Returns:
        The full path vertices (transmitter, intermediate bounces,
        receiver).
    """
    from differt.em import InteractionType  # ruff: ignore[import-outside-top-level]

    num_path_candidates, order = path_candidates.shape

    active = path_candidates >= 0
    safe = jnp.where(active, path_candidates, 0)
    is_diffraction = active & (interaction_types == InteractionType.DIFFRACTION)
    is_bending = active & (
        is_diffraction
        | (interaction_types == InteractionType.REFLECTION)
        | (interaction_types == InteractionType.SCATTERING)
    )

    if use_fermat:
        object_origin, object_vectors = _bending_geometry_for_fermat(
            mesh, safe, is_diffraction
        )
    else:
        object_origin, object_normal = _surface_geometry(mesh, safe)

    if needs_splice:
        perm = _bending_first_permutation(is_bending)
        object_origin = jnp.take_along_axis(object_origin, perm[..., None], axis=-2)
        is_bending_for_solve = jnp.take_along_axis(is_bending, perm, axis=-1)
        if use_fermat:
            object_vectors = jnp.take_along_axis(
                object_vectors, perm[..., None, None], axis=-3
            )
        else:
            object_normal = jnp.take_along_axis(object_normal, perm[..., None], axis=-2)
    else:
        is_bending_for_solve = is_bending

    # Collapse non-bending slots (TRANSMISSION and placeholders, now a
    # trailing suffix after reordering) to the receiver plane, exactly as
    # done for trailing '-1' placeholders today: both the image (forward
    # pass) and the intersection point (backward pass) of the recursion
    # then collapse to the receiver itself.
    object_origin_for_solve = jnp.where(
        is_bending_for_solve[None, ..., None],
        object_origin[None, ...],
        rx_vertices[:, None, None, :],
    )
    if use_fermat:
        object_vectors_for_solve = jnp.where(
            is_bending_for_solve[None, ..., None, None],
            object_vectors[None, ...],
            jnp.zeros_like(object_vectors)[None, ...],
        )
        solved = fermat_path_on_linear_objects(
            tx_vertices[:, None, None, :],
            rx_vertices[None, :, None, :],
            object_origin_for_solve,
            object_vectors_for_solve,
            **(fermat_kwargs or {}),
        )
    else:
        object_normal_for_solve = jnp.where(
            is_bending_for_solve[None, ..., None],
            object_normal[None, ...],
            jnp.zeros_like(object_normal)[None, ...],
        )
        solved = image_method(
            tx_vertices[:, None, None, :],
            rx_vertices[None, :, None, :],
            object_origin_for_solve,
            object_normal_for_solve,
        )

    if needs_splice:
        inv_perm = jnp.argsort(perm, axis=-1)
        inv_perm = jnp.broadcast_to(inv_perm, (*solved.shape[:-2], order))
        solved = jnp.take_along_axis(solved, inv_perm[..., None], axis=-2)

    full_paths = assemble_path(
        tx_vertices[:, None, None, :], solved, rx_vertices[None, :, None, :]
    )

    if not needs_splice:
        return full_paths

    is_transmission = active & (interaction_types == InteractionType.TRANSMISSION)
    is_real = jnp.concatenate(
        (
            jnp.ones((num_path_candidates, 1), dtype=bool),
            is_bending,
            jnp.ones((num_path_candidates, 1), dtype=bool),
        ),
        axis=-1,
    )
    prev_idx, next_idx = _nearest_real_neighbor_indices(is_real)
    prev_idx = jnp.broadcast_to(prev_idx, (*full_paths.shape[:-2], order + 2))
    next_idx = jnp.broadcast_to(next_idx, (*full_paths.shape[:-2], order + 2))
    prev_pos = jnp.take_along_axis(full_paths, prev_idx[..., None], axis=-2)
    next_pos = jnp.take_along_axis(full_paths, next_idx[..., None], axis=-2)

    plane_vertex, plane_normal = _surface_geometry(mesh, safe)
    transmit_pos = intersection_of_ray_with_plane(
        prev_pos[..., 1:-1, :],
        next_pos[..., 1:-1, :] - prev_pos[..., 1:-1, :],
        plane_vertex,
        plane_normal,
    )
    transmit_pos = jnp.concatenate(
        (
            full_paths[..., :1, :],
            transmit_pos,
            full_paths[..., -1:, :],
        ),
        axis=-2,
    )

    is_transmission_full = jnp.concatenate(
        (
            jnp.zeros((num_path_candidates, 1), dtype=bool),
            is_transmission,
            jnp.zeros((num_path_candidates, 1), dtype=bool),
        ),
        axis=-1,
    )
    return jnp.where(is_transmission_full[None, ..., None], transmit_pos, full_paths)

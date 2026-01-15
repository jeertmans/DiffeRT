import math

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped

from differt.rt import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    rays_intersect_any_triangle,
    rays_intersect_triangles,
)
from differt.scene import TriangleScene


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def reward(
    predicted_path_candidates: Int[Array, "*batch order"],
    scene: TriangleScene,
) -> Float[Array, "*batch"]:
    """
    Reward predicted path candidates depending on whether it
    produces a valid path in the given scene.

    Args:
        predicted_path_candidates: The path candidates to evaluate.
        scene: The scene on which to evaluate the path candidates.

    Returns:
        A reward of 0 or 1 for each path candidate.
    """
    *batch, order = predicted_path_candidates.shape
    p = math.prod(batch)
    if p == 0:
        return jnp.zeros(batch)
    r = scene.compute_paths(
        path_candidates=predicted_path_candidates.reshape(p, order)
    ).mask.astype(float)

    return r.reshape(*batch)


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def decreasing_edge_reward(
    predicted_path_candidates: Int[Array, "*batch order"],
    scene: TriangleScene,
    gamma: float = 1.0,
) -> Float[Array, "*batch"]:
    """
    Reward predicted path candidates giving credit to each valid edge,
    with a decreasing factor gamma.

    The reward is calculated as sum(gamma**i) for i in 0..order,
    where term i is included only if the path is valid up to that segment.

    Args:
        predicted_path_candidates: The path candidates to evaluate.
        scene: The scene on which to evaluate the path candidates.
        gamma: The discount factor for the reward.

    Returns:
        A cumulative reward for each path candidate.
    """
    *batch, order = predicted_path_candidates.shape
    p = math.prod(batch)
    if p == 0:
        return jnp.zeros(batch)

    # Reshape for computation
    path_candidates = predicted_path_candidates.reshape(p, order)

    # Compute full paths (vertices)
    # We do NOT use the mask from compute_paths because we want partial validity.
    paths = scene.compute_paths(path_candidates=path_candidates)

    # Extract components for validity checks
    # paths.vertices shape: [p, order+2, 3] (Tx, P1, ..., Pk, Rx)
    # paths.objects shape: [p, order+2, 1] (Indices of Tx, Tri1, ..., TriK, Rx)

    full_paths = paths.vertices

    # Get triangle vertices for strict intersection checks
    # The 'compute_paths' logic internally gets 'triangle_vertices' corresponding to path candidates.
    # We need to replicate that extraction to use 'rays_intersect_triangles'.

    # From _triangle_scene.py (lines 86-93):
    mesh = scene.mesh
    k = (
        2 if mesh.assume_quads else 1
    )  # Assuming k=1 for simplicity or reading from scene?
    # Actually, we should handle assume_quads.

    # But wait, scene.compute_paths does all this.
    # Can we just implement the checks using the coordinates?

    # Gather triangle vertices corresponding to the path candidates
    # path_candidates indices point to mesh.triangles

    if mesh.assume_quads:
        # [p 2*order]
        pc_expanded = jnp.repeat(path_candidates, 2, axis=-1)
        pc_expanded = pc_expanded.at[..., 1::2].add(1)
        k_factor = 2
    else:
        pc_expanded = path_candidates
        k_factor = 1

    # [p k*order 3]
    tri_indices = jnp.take(mesh.triangles, pc_expanded, axis=0).reshape(
        p, k_factor * order, 3
    )
    # [p k*order 3 3]
    tri_vertices = jnp.take(mesh.vertices, tri_indices, axis=0).reshape(
        p, k_factor * order, 3, 3
    )

    # Components to check:
    ray_origins = full_paths[..., :-1, :]  # [p, order+1, 3]
    ray_directions = jnp.diff(full_paths, axis=-2)  # [p, order+1, 3]

    # 1. Check if interaction points are inside their respective triangles
    # Points P1...Pk correspond to rays 0...k-1 hitting triangles 0...k-1
    # We check if ray_origins[..., :-1, :] + ray_directions[..., :-1, :] hits the triangles.
    # Actually, P_i = ray_origin_i + ray_direction_i

    # rays_intersect_triangles checks if the ray intersects.
    # usage: rays_intersect_triangles(origins, directions, triangle_vertices)

    # We only care about the first 'order' segments for triangle intersection.
    # The last segment (to Rx) does not hit a triangle (it hits Rx).

    if mesh.assume_quads:
        # Repeat rays matching the quad structure
        ro_rep = jnp.repeat(ray_origins[..., :-1, :], 2, axis=-2)
        rd_rep = jnp.repeat(ray_directions[..., :-1, :], 2, axis=-2)

        inside_triangles = (
            rays_intersect_triangles(
                ro_rep, rd_rep, tri_vertices, epsilon=None
            )[1]
            .reshape(p, order, 2)
            .any(
                axis=-1
            )  # Inside at least one of the two triangles of the quad
            .all(
                axis=-1
            )  # This mimics 'all' from original, but we want partial?
            # Wait, we want mask PER INTERACTION.
        )
        # Actually we want per-interaction validity.
        inside_triangles = (
            rays_intersect_triangles(
                ro_rep, rd_rep, tri_vertices, epsilon=None
            )[1]
            .reshape(p, order, 2)
            .any(axis=-1)  # [p, order] boolean
        )
    else:
        inside_triangles = rays_intersect_triangles(
            ray_origins[..., :-1, :],
            ray_directions[..., :-1, :],
            tri_vertices,
            epsilon=None,
        )[1]  # [p, order] boolean

    # 2. Check valid reflections (front/back side)
    # Mirror vertices needed.
    # mirror_vertices = tri_vertices[..., 0, :] (one vtx per tri)
    # But with quads/k_factor it's tricky.
    # Simplified: extraction logic from _triangle_scene.py

    # [p order 3]
    # We take the normals from the mesh
    # path_candidates points to triangles.
    # But assume_quads affects WHICH triangle index we use for normal?
    # Usually normals are same for quad pair (if properly made).
    # _triangle_scene.py takes every 2nd if assuming quads.

    cand_indices = path_candidates
    if mesh.assume_quads:
        # The indices in path_candidates refer to "quads" (pairs of triangles) if we strictly follow how compute_paths interprets them?
        # WAIT. compute_paths logic:
        # path_candidates = jnp.repeat(path_candidates, 2, axis=-1)
        # So input 'path_candidates' are indices of Quads (if assume_quads is True)?
        # Or indices of first triangle of quad?
        # Usually 'path_candidates' in Image Method are indices of primitives.
        # If 'assume_quads' is True, the input indices are multiplied/processed.

        # Let's trust _triangle_scene.py extraction:
        # mirror_vertices = triangle_vertices[..., ::2, 0, :]
        pass

    # For extraction of normals:
    # mirror_normals = jnp.take(mesh.normals, path_candidates[..., ::2], axis=0) (if quads)

    # Let's rebuild mirrors
    if mesh.assume_quads:
        mirror_normals = jnp.take(
            mesh.normals, path_candidates, axis=0
        )  # path_candidates is quad index?
        # In _triangle_scene:
        # path_candidates (input) is [p, order]
        # expanded: jnp.repeat(..., 2) and shift.
        # mirror_normals extraction uses original path_candidates?
        # "jnp.take(mesh.normals, path_candidates[..., ::(2 if ...)], axis=0)"
        # Use simple logic:
        mirror_normals = jnp.take(
            mesh.normals, path_candidates, axis=0
        )  # [p, order, 3]

        # Mirror vertices: just pick one from the tri vertices we got.
        # tri_vertices was [p, 2*order, 3, 3]
        mirror_vertices = tri_vertices[..., ::2, 0, :]  # [p, order, 3]
    else:
        mirror_normals = jnp.take(mesh.normals, path_candidates, axis=0)
        mirror_vertices = tri_vertices[..., 0, :]

    valid_reflections = consecutive_vertices_are_on_same_side_of_mirrors(
        full_paths, mirror_vertices, mirror_normals
    )  # [p, order] boolean

    # 3. Check blocked
    # This is for ALL segments (order+1).
    # rays_intersect_any_triangle returns [p, order+1] boolean (if Reduce=False?)
    # Wait, rays_intersect_any_triangle by default might reduce?
    # In _triangle_scene, it does `.any(axis=-1)` (reduce on *triangles* it hit?)
    # No.
    # rays_intersect_any_triangle checks if ray intersects *any* triangle in the scene (blockage).
    # It returns shape matching ray shape.

    blocked = rays_intersect_any_triangle(
        ray_origins,
        ray_directions,
        mesh.triangle_vertices,
        active_triangles=mesh.mask,
        epsilon=None,  # Default
        hit_tol=None,
    )  # [p, order+1] boolean (True if blocked)

    # Combine masks
    # We want validity.
    # Validity at step i (0..order-1) means:
    # - Segment i (Tx->P1 or Pi->Pi+1) is unblocked. (blocked[..., i] is False)
    # - Point Pi+1 is valid (inside_triangles[..., i] & valid_reflections[..., i])
    # AND previous steps valid.

    # Last segment (Pk->Rx) is index 'order'.
    # It only needs to be unblocked.

    # Let's accumulate validity.

    valid_mask = jnp.ones((p,), dtype=float)
    cumulative_reward = jnp.zeros((p,), dtype=float)

    # Loop over steps
    for i in range(order):
        # Step i involves:
        # 1. Traversing segment i (from P_i to P_{i+1})
        # 2. Hitting P_{i+1} correctly.

        # Segment i blocked?
        seg_clear = ~blocked[..., i]

        # Interaction i valid?
        interaction_ok = inside_triangles[..., i] & valid_reflections[..., i]

        step_ok = seg_clear & interaction_ok

        # Update validity chain
        valid_mask = valid_mask * step_ok.astype(float)

        # Add reward
        cumulative_reward += valid_mask * (gamma**i)

    # Final segment check (to Rx)
    # If we are valid up to Pk, we try to reach Rx.
    i = order
    seg_clear = ~blocked[..., i]

    valid_mask = valid_mask * seg_clear.astype(float)
    cumulative_reward += valid_mask * (gamma**i)

    return cumulative_reward.reshape(*batch)


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def accuracy(
    scene: TriangleScene,
    predicted_path_candidates: Int[Array, "num_path_candidates order"],
) -> Float[Array, " "]:
    if predicted_path_candidates.shape[0] == 0:
        return jnp.zeros(())
    paths = scene.compute_paths(path_candidates=predicted_path_candidates)
    return paths.mask.astype(float).mean()


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def hit_rate(
    scene: TriangleScene,
    predicted_path_candidates: Int[Array, "num_path_candidates order"],
) -> Float[Array, " "]:
    num_path_candidates, order = predicted_path_candidates.shape
    if num_path_candidates == 0:
        return jnp.zeros(())
    paths = scene.compute_paths(path_candidates=predicted_path_candidates)
    num_paths_found = paths.mask_duplicate_objects().mask.astype(float).sum()
    num_paths_total = scene.compute_paths(order=order).mask.astype(float).sum()
    no_valid_paths = num_paths_total == 0
    num_paths_total = jnp.where(no_valid_paths, 1.0, num_paths_total)
    return jnp.where(no_valid_paths, 1.0, num_paths_found / num_paths_total)

import math

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from differt.scene import TriangleScene


def reward_fn(
    predicted_path_candidates: Int[Array, "*batch order"],
    scene: TriangleScene,
) -> Float[Array, "*batch"]:
    """
    Reward predicted path candidates depending on whether it produces a valid path in the given scene.

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
    r = scene.compute_paths(  # type: ignore[possibly-missing-attribute]
        path_candidates=predicted_path_candidates.reshape(p, order)
    ).mask.astype(float)

    return r.reshape(*batch)


def accuracy(
    scene: TriangleScene,
    predicted_path_candidates: Int[Array, "num_path_candidates order"],
) -> Float[Array, " "]:
    """
    Compute the accuracy of the predicted path candidates on the given scene.

    Args:
        scene: The scene on which to evaluate the path candidates.
        predicted_path_candidates: The path candidates to evaluate.

    Returns:
        The accuracy of the predicted path candidates.
    """
    if predicted_path_candidates.shape[0] == 0:
        return jnp.zeros(())
    paths = scene.compute_paths(path_candidates=predicted_path_candidates)
    return paths.mask.astype(float).mean()  # type: ignore[possibly-missing-attribute]


def hit_rate(
    scene: TriangleScene,
    predicted_path_candidates: Int[Array, "num_path_candidates order"],
) -> Float[Array, " "]:
    """
    Compute the hit rate of the predicted path candidates on the given scene.

    Args:
        scene: The scene on which to evaluate the path candidates.
        predicted_path_candidates: The path candidates to evaluate.

    Returns:
        The hit rate of the predicted path candidates.
        On scenes with no valid paths, returns 1.0.
    """
    num_path_candidates, order = predicted_path_candidates.shape
    if num_path_candidates == 0:
        return jnp.zeros(())
    paths = scene.compute_paths(path_candidates=predicted_path_candidates)
    num_paths_found = paths.mask_duplicate_objects().mask.astype(float).sum()
    num_paths_total = scene.compute_paths(order=order).mask.astype(float).sum()  # type: ignore[possibly-missing-attribute]
    no_valid_paths = num_paths_total == 0
    num_paths_total = jnp.where(no_valid_paths, 1.0, num_paths_total)
    return jnp.where(no_valid_paths, 1.0, num_paths_found / num_paths_total)

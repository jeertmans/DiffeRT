import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from differt.scene import TriangleScene


@eqx.filter_jit
def reward(
    predicted_path_candidates: Int[Array, "batch order"],
    scene: TriangleScene,
    *,
    non_differentiable: bool = False,
) -> Float[Array, " batch"]:
    """
    Reward predicted path candidates depending on whether it
    produces a valid path in the given scene.

    Args:
        predicted_path_candidates: The path candidates to evaluate.
        scene: The scene on which to evaluate the path candidates.
        non_differentiable: Whether to stop gradients.

    Returns:
        A reward of 0 or 1 for each path candidate.
    """
    r = scene.compute_paths(
        path_candidates=predicted_path_candidates
    ).mask.astype(jnp.float32)
    if non_differentiable:
        return jax.lax.stop_gradient(r)

    return r


@eqx.filter_jit
def accuracy(
    scene: TriangleScene,
    predicted_path_candidates: Int[Array, "num_path_candidates order"],
) -> Float[Array, " "]:
    paths = scene.compute_paths(path_candidates=predicted_path_candidates)
    num_valid_paths = paths.mask.astype(jnp.float32).sum()
    return num_valid_paths / predicted_path_candidates.shape[0]


eqx.filter_jit


def hit_rate(
    scene: TriangleScene,
    predicted_path_candidates: Int[Array, "num_path_candidates order"],
) -> Float[Array, " "]:
    _, order = predicted_path_candidates.shape
    paths = scene.compute_paths(path_candidates=predicted_path_candidates)
    num_paths_found = (
        paths.mask_duplicate_objects().mask.astype(jnp.float32).sum()
    )
    num_paths_total = (
        scene.compute_paths(order=order).mask.astype(jnp.float32).sum()
    )
    return num_paths_found / num_paths_total

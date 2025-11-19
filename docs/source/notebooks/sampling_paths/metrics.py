import equinox as eqx
import jax
from jaxtyping import Array, Float, Int

from differt.scene import TriangleScene


@eqx.filter_jit
def reward(
    path_candidates: Int[Array, "batch order"],
    scene: TriangleScene,
    *,
    non_differentiable: bool = False,
) -> Float[Array, " batch"]:
    """
    Reward predicted path candidates depending on whether it
    produces a valid path in the given scene.

    Args:
        path_candidates: The path candidates to evaluate.
        scene: The scene on which to evaluate the path candidates.
        non_differentiable: Whether to stop gradients.

    Returns:
        A reward of 0 or 1 for each path candidate.
    """
    r = scene.compute_paths(path_candidates=path_candidates).mask.astype(float)
    if non_differentiable:
        return jax.lax.stop_gradient(r)

    return r

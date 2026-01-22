import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray

from differt.scene import TriangleScene

from .agent import Agent


class TestAgent:
    @pytest.mark.parametrize("degenerate", [False, True])
    @jax.debug_infs(True)
    def test_train(
        self,
        degenerate: bool,
        agent: Agent,
        scene: TriangleScene,
        key: PRNGKeyArray,
    ) -> None:
        # We check that we can overfit to a single, simple scene and learn to avoid invalid paths
        train_key, eval_key = jr.split(key)

        loss = jnp.inf
        num_episodes = {1: 3000, 2: 15_000, 3: 15_000}[agent.model.order]

        if degenerate:
            # Make the scene degenerate by masking all triangles
            def random_scene(*, key: PRNGKeyArray) -> TriangleScene:
                del key
                return eqx.tree_at(
                    lambda x: x.mesh.mask,
                    scene,
                    jnp.zeros_like(scene.mesh.mask),  # type: ignore[invalid-argument-type]
                )

        else:

            def random_scene(*, key: PRNGKeyArray) -> TriangleScene:
                return eqx.tree_at(lambda x: x.mesh, scene, scene.mesh.shuffle(key=key))

        agent = eqx.tree_at(lambda a: a.scene_fn, agent, random_scene)

        for _ in range(num_episodes):
            key, scene_key, train_key = jr.split(key, 3)
            agent, loss = agent.train(scene_key=scene_key, key=train_key)

        if degenerate:
            return  # Can't sample valid paths in degenerate scene

        path_candidates = jax.vmap(
            lambda key: agent.model(scene, inference=True, key=key)
        )(jr.split(eval_key, 100))

        paths = scene.compute_paths(path_candidates=path_candidates)
        valid_paths = scene.compute_paths(order=agent.model.order).masked()

        if not paths.mask.all():  # type: ignore[possibly-missing-attribute]
            invalid_paths = path_candidates[~paths.mask, :]  # type: ignore[unsupported-operator]
            per_invalid = 100 * invalid_paths.shape[0] / path_candidates.shape[0]
            msg = (
                f"Agent trained model (final loss: {loss:.2e}) "
                "but generated path candidates still contain invalid paths "
                f"({per_invalid:.2f}% of them):\n{invalid_paths}, "
                f"but valid are:\n{valid_paths.objects[:, 1:-1]}"
            )
            raise AssertionError(msg)

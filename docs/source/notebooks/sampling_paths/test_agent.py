import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from differt.scene import TriangleScene

from .agent import Agent


class TestAgent:
    def test_train(
        self, agent: Agent, scene: TriangleScene, key: PRNGKeyArray
    ) -> None:
        # We check that we can overfit to a single scene and learn to avoid unreachable objects
        train_key, eval_key = jr.split(key)

        loss = jnp.inf
        num_steps = 0
        num_max_steps = 1_000

        while num_steps < num_max_steps:
            key, shuffle_key, train_key = jr.split(key, 3)
            train_scene = eqx.tree_at(
                lambda x: x.mesh, scene, scene.mesh.shuffle(key=shuffle_key)
            )
            agent, loss = agent.train(scene=train_scene, key=train_key)
            num_steps += 1

        unreachable_objects = jnp.array([0, 3])
        path_candidates = jax.vmap(
            lambda key: agent.model(scene, inference=True, key=key)
        )(jr.split(eval_key, 10))

        paths = scene.compute_paths(order=agent.model.order)
        valid_paths = paths.masked()

        assert not jnp.isin(path_candidates, unreachable_objects).any(), (
            f"Path candidates should not contain unreachable objects, got: {path_candidates} (accepted are: {valid_paths.objects[:, 1:-1]})"
        )

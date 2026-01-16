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

        while loss > 1e-3 and num_steps < num_max_steps:
            key, train_key = jr.split(key)
            agent, loss, avg_reward = agent.train(scene=scene, key=key)
            num_steps += 1

        # if num_steps == num_max_steps:
        #     pytest.fail(
        #         f"Agent did not converge within {num_max_steps} steps. "
        #         f"Final loss: {loss}, average reward: {avg_reward}"
        #     )

        unreachable_objects = jnp.array([0, 3])
        path_candidates = jax.vmap(
            lambda key: agent.model(scene, inference=True, key=key)
        )(jr.split(eval_key, 10))

        paths = scene.compute_paths(order=agent.model.order)
        valid_paths = paths.masked()

        assert not jnp.isin(path_candidates, unreachable_objects).any(), (
            f"Path candidates should not contain unreachable objects, got: {path_candidates} (accepted are: {valid_paths.objects[:, 1:-1]})"
        )

        assert loss < 1.0

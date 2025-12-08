import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    Int,
    Key,
    PRNGKeyArray,
    jaxtyped,
)

from beartype import beartype as typechecker

from differt.scene import (
    TriangleScene,
)

from .generators import random_scene
from .metrics import accuracy, hit_rate
from .submodels import Flow


class Model(eqx.Module):
    order: int = eqx.field(static=True)

    flow: Flow

    def __init__(
        self,
        *,
        order: int,
        num_embeddings: int,
        width_size: int,
        depth: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ) -> None:
        self.order = order

        self.flow = Flow(
            order=order,
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            dropout_rate=dropout_rate,
            key=key,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        scene: TriangleScene,
        *,
        inference: bool | None = None,
        key: PRNGKeyArray,
    ) -> Int[Array, " order"]:
        partial_path_candidate = -jnp.ones(self.order, dtype=int)
        parent_flow_key, key = jr.split(key)
        parent_flows = self.flow(
            scene,
            partial_path_candidate,
            key=parent_flow_key,
        )

        for i, key in enumerate(jr.split(key, self.order)):
            edge_flow_key, action_key = jr.split(key)

            action = jr.categorical(action_key, logits=jnp.log(parent_flows))
            partial_path_candidate = partial_path_candidate.at[i].set(action)

            edge_flows = self.flow(
                scene,
                partial_path_candidate,
                inference=inference,
                key=edge_flow_key,
            )

            parent_flows = edge_flows

        return partial_path_candidate

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def evaluate(
        self,
        scene_keys: Key[Array, " num_scenes"],
        *,
        num_path_candidates: int = 1,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, " "], Float[Array, " "]]:
        """
        Evaluate the model accuracy and hit rate on a sequence of scenes.

        Args:
            scene_keys: The scene keys to generate scenes on which to evaluate the model.
            num_path_candidates: The number of path candidates that will be
                generated to compute the accuracy.
            key: The random key to be used.

        Returns:
            The average accuracy and average hit rate.
        """

        def _evaluate(
            scene_key: PRNGKeyArray, key: PRNGKeyArray
        ) -> tuple[Float[Array, " "], Float[Array, " "]]:
            scene = random_scene(key=scene_key)
            path_candidates = jax.vmap(
                lambda key: self(scene, inference=True, key=key)
            )(jr.split(key, num_path_candidates))
            return accuracy(scene, path_candidates), hit_rate(
                scene, path_candidates
            )

        num_scenes = scene_keys.shape[0]
        keys = jr.split(key, num_scenes)

        accuracies, hit_rates = jax.vmap(_evaluate)(scene_keys, keys)

        return accuracies.mean(), jnp.nanmean(hit_rates)

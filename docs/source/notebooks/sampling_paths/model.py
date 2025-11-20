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
)

from differt.scene import (
    TriangleScene,
)

from .generators import random_scene
from .metrics import accuracy, hit_rate
from .submodels import Flow, Z


class Model(eqx.Module):
    order: int = eqx.field(static=True)

    shared: eqx.nn.Shared

    def __init__(
        self,
        *,
        order: int,
        num_embeddings: int,
        width_size: int,
        depth: int,
        key: PRNGKeyArray,
    ) -> None:
        self.order = order

        flow_key, z_key = jr.split(key)
        flow = Flow(
            order=order,
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=flow_key,
        )
        z = Z(
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=z_key,
        )

        # Explicitly share the scene encoder
        where = lambda flow_and_z: flow_and_z[1].scene_encoder
        get = lambda flow_and_z: flow_and_z[0].scene_encoder
        self.shared = eqx.nn.Shared((flow, z), where, get)

    @property
    def flow(self) -> Flow:
        return self.shared()[0]

    @property
    def Z(self) -> Flow:
        return self.shared()[1]

    @eqx.filter_jit
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

            logits = parent_flows
            action = jr.categorical(action_key, logits=logits)
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
            model: The model to evaluate.
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

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
    Int,
    Key,
    PRNGKeyArray,
)

from differt.scene import TriangleScene

from .generators import random_scene
from .metrics import accuracy, hit_rate, reward_fn
from .model import Model

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


def flow_matching_loss(
    model: Model,
    scene: TriangleScene,
    key: PRNGKeyArray,
) -> Float[Array, " "]:
    """
    Compute the flow matching loss for a given model and scene.
    """
    path_candidate, flows = model(scene, inference=False, key=key)
    parent_flows = jnp.take(flows, path_candidate, axis=0)
    sum_edge_flows = flows.sum(axis=0)
    sum_edge_flows = jnp.roll(sum_edge_flows, -1)
    sum_edge_flows = sum_edge_flows.at[-1].set(reward_fn(path_candidate, scene))
    flows_mismatch = (parent_flows - sum_edge_flows) ** 2
    return flows_mismatch.sum()


def loss(
    model: Model,
    scene: TriangleScene,
    batch_size: int,
    key: PRNGKeyArray,
) -> Float[Array, " "]:
    loss_values = jax.vmap(flow_matching_loss, in_axes=(None, None, 0))(
        model, scene, jr.split(key, batch_size)
    )
    return loss_values.mean()


class Agent(eqx.Module):
    """
    Agent that trains a model to learn to sample path candidates on triangle scenes.

    It uses flow matching to learn a flow model that estimates the flow of paths
    through the scene, and uses this flow model to sample path candidates.
    """

    # Static
    batch_size: int = eqx.field(static=True)

    # Trainable
    model: Model

    # Updatable
    optim: optax.GradientTransformationExtraArgs
    opt_state: optax.OptState
    steps_count: Int[Array, " "]

    def __init__(
        self,
        *,
        model: Model,
        batch_size: int = 64,
        optim: optax.GradientTransformationExtraArgs | None = None,
        epsilon: Float[ArrayLike, ""] = 0.9,
        delta_epsilon: Float[ArrayLike, ""] = 1e-5,
        min_epsilon: Float[ArrayLike, ""] = 0.1,
    ) -> None:
        self.batch_size = batch_size
        self.model = model

        self.optim = optax.adam(3e-5) if optim is None else optim
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.steps_count = jnp.array(0)

    @eqx.filter_jit
    def train(
        self,
        scene: TriangleScene,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Self, Float[Array, " "]]:
        """
        Train the model on one scene using the flow matching loss.

        Args:
            scene: The scene to train on.
            key: The key to use for training.

        Returns:
            The updated agent and the average loss value.
        """

        loss_value, grads = eqx.filter_value_and_grad(loss)(
            self.model,
            scene,
            batch_size=self.batch_size,
            key=key,
        )

        updates, opt_state = self.optim.update(
            grads, self.opt_state, eqx.filter(self.model, eqx.is_array)
        )

        return (
            eqx.tree_at(
                lambda agent: (
                    agent.model,
                    agent.opt_state,
                    agent.steps_count,
                ),
                self,
                (
                    eqx.apply_updates(self.model, updates),
                    opt_state,
                    self.steps_count + 1,
                ),
            ),
            loss_value,
        )

    @eqx.filter_jit
    def evaluate(
        self,
        scene_keys: Key[Array, " num_scenes"],
        *,
        num_path_candidates: int = 10,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, " "], Float[Array, " "]]:
        """
        Evaluate the model accuracy and hit rate on a sequence of scenes.

        Args:
            scene_keys: The scene keys to generate scenes on which to evaluate the model.
            num_path_candidates: The number of path candidates that will be
                generated to compute the accuracy and hit rate.
            key: The random key to be used.

        Returns:
            The average accuracy and average hit rate.
        """

        def _evaluate(
            scene_key: PRNGKeyArray, key: PRNGKeyArray
        ) -> tuple[Float[Array, " "], Float[Array, " "]]:
            scene = random_scene(key=scene_key)
            path_candidates = jax.vmap(
                lambda key: self.model(scene, inference=True, key=key)
            )(jr.split(key, num_path_candidates))
            return accuracy(scene, path_candidates), hit_rate(
                scene, path_candidates
            )

        num_scenes = scene_keys.shape[0]
        keys = jr.split(key, num_scenes)

        accuracies, hit_rates = jax.vmap(_evaluate)(scene_keys, keys)

        return accuracies.mean(), jnp.nanmean(hit_rates)

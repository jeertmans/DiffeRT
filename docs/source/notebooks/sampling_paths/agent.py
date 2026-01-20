from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import (
    Array,
    Float,
    Int,
    Key,
    PRNGKeyArray,
)

from differt.scene import TriangleScene

from .generators import random_scene
from .metrics import accuracy, hit_rate
from .model import Model

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


def batch_loss(
    model: Model,
    scene: TriangleScene,
    batch_size: int,
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    _, loss_values = jax.vmap(
        lambda key: model(scene, inference=False, key=key)
    )(jr.split(key, batch_size))
    return loss_values.mean()


def decrease_epsilon(
    delta_epsilon: float, min_epsilon: float
) -> optax.GradientTransformation:
    def update_fn(
        updates: Model, state: optax.OptState, params: Model
    ) -> tuple[Model, optax.OptState]:
        update_epsilon = -jnp.minimum(
            delta_epsilon, params.epsilon - min_epsilon
        )
        updates = eqx.tree_at(lambda m: m.epsilon, updates, update_epsilon)
        return updates, state

    return optax.GradientTransformation(
        optax.init_empty_state,
        update_fn,
    )


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
    steps_count: Int[Array, ""]

    def __init__(
        self,
        *,
        model: Model,
        batch_size: int = 64,
        optim: optax.GradientTransformationExtraArgs | None = None,
        delta_epsilon: float = 1e-5,
        min_epsilon: float = 0.1,
    ) -> None:
        self.batch_size = batch_size
        self.model = model

        optim = optax.adam(3e-5) if optim is None else optim

        self.optim = optax.chain(
            optim, decrease_epsilon(delta_epsilon, min_epsilon)
        )
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.steps_count = jnp.array(0)

    @eqx.filter_jit
    def train(
        self,
        scene: TriangleScene,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Self, Float[Array, ""]]:
        """
        Train the model on one scene using the flow matching loss.

        Args:
            scene: The scene to train on.
            key: The key to use for training.

        Returns:
            The updated agent and the average loss value.
        """

        loss_value, grads = eqx.filter_value_and_grad(batch_loss)(
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
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
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
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
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

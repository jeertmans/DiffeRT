from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

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
from .replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


class SceneFn(Protocol):
    def __call__(self, *, key: PRNGKeyArray) -> TriangleScene: ...


def batch_loss(
    model: Model,
    scene_key: PRNGKeyArray,
    keys: Key[Array, " batch_size"],
    *,
    scene_fn: SceneFn,
    debug: bool = False,
) -> tuple[
    Float[Array, ""],
    tuple[Int[Array, "batch_size order"], Float[Array, " batch_size"]],
]:
    if debug:
        path_candidates, loss_values, rewards = jax.lax.map(
            lambda key: model(scene_fn(key=scene_key), inference=False, key=key),
            keys,
        )
    else:
        path_candidates, loss_values, rewards = jax.vmap(
            lambda key: model(scene_fn(key=scene_key), inference=False, key=key)
        )(keys)

    return loss_values.mean(), (path_candidates, rewards)


@jax.debug_nans(False)
@jax.debug_infs(False)
def replay_loss(
    model: Model,
    scene_keys: Key[Array, " batch_size"],
    path_candidates: Int[Array, " batch_size order"],
    rewards: Float[Array, " batch_size"],
    *,
    scene_fn: SceneFn,
    debug: bool = False,
) -> Float[Array, ""]:
    if debug:
        _, loss_values, _ = jax.lax.map(
            lambda scene_key, path_candidate: model(
                scene_fn(key=scene_key),
                replay=path_candidate,
                inference=False,
                key=jr.key(0),
            ),
            (scene_keys, path_candidates),
        )
    else:
        _, loss_values, _ = jax.vmap(
            lambda scene_key, path_candidate: model(
                scene_fn(key=scene_key),
                replay=path_candidate,
                inference=False,
                key=jr.key(0),
            )
        )(scene_keys, path_candidates)
    return jnp.sum(loss_values * rewards)


def decrease_epsilon(
    delta_epsilon: float, min_epsilon: float
) -> optax.GradientTransformation:
    """
    Custom Optax transformation that decreases the epsilon parameter of the model by delta_epsilon at each optimization step, down to a minimum of min_epsilon.

    Args:
        delta_epsilon: The amount by which to decrease epsilon at each step.
        min_epsilon: The minimum value that epsilon can take.

    Returns:
        An Optax GradientTransformation that decreases epsilon.
    """

    def update_fn(
        updates: Model, state: optax.OptState, params: Model
    ) -> tuple[Model, optax.OptState]:
        update_epsilon = -jnp.minimum(delta_epsilon, params.epsilon - min_epsilon)
        updates = eqx.tree_at(lambda m: m.epsilon, updates, update_epsilon)
        return updates, state

    return optax.GradientTransformation(
        optax.init_empty_state,
        update_fn,
    )


class Agent(eqx.Module):
    """
    Agent that trains a model to learn to sample path candidates on triangle scenes.

    It uses flow matching to learn a flow model that estimates the flow of paths through the scene, and uses this flow model to sample path candidates.

    If a replay buffer is provided, it will store successful experiences and use them for training.
    """

    # Static
    batch_size: int = eqx.field(static=True)
    # Static but can be changed
    scene_fn: SceneFn
    debug: bool
    # Learned
    model: Model
    # Training
    optim: optax.GradientTransformationExtraArgs
    opt_state: optax.OptState
    steps_count: Int[Array, ""]
    replay_buffer: ReplayBuffer | None
    """Optional replay buffer to store successful experiences and use them for training."""

    def __init__(
        self,
        *,
        model: Model,
        batch_size: int = 64,
        optim: optax.GradientTransformationExtraArgs | None = None,
        delta_epsilon: float = 1e-5,
        min_epsilon: float = 0.1,
        replay_buffer_capacity: int | None = 10_000,
        scene_fn: SceneFn = random_scene,
    ) -> None:
        self.batch_size = batch_size
        self.scene_fn = scene_fn
        self.debug = model.debug

        self.model = model

        optim = optax.adam(3e-5) if optim is None else optim
        self.optim = optax.chain(optim, decrease_epsilon(delta_epsilon, min_epsilon))
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.steps_count = jnp.array(0)
        self.replay_buffer = (
            ReplayBuffer(
                replay_buffer_capacity,
                order=model.order,
                scene_key_dtype=jr.key(0).dtype,
            )
            if replay_buffer_capacity is not None
            else None
        )

    def set_debug_mode(self, debug: bool) -> Self:
        """
        Set the debug mode of the model.

        Args:
            debug: Whether to enable debug mode.

        Returns:
            The updated agent with the debug mode set.
        """
        return eqx.tree_at(
            lambda agent: (agent.debug, agent.model.debug),
            self,
            (debug, debug),
        )

    @eqx.filter_jit
    def train(
        self,
        scene_key: PRNGKeyArray,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Self, Float[Array, ""]]:
        """
        Train the model on one scene using the flow matching loss.

        Args:
            scene_key: The key to generate the scene to train on.
            key: The key to use for training.

        Returns:
            The updated agent and the average loss value.
        """
        keys = jr.split(key, self.batch_size)

        # 1st train step: flow matching

        (loss_value, (path_candidates, rewards)), grads = eqx.filter_value_and_grad(
            partial(batch_loss, scene_fn=self.scene_fn, debug=self.debug),
            has_aux=True,
        )(
            self.model,
            scene_key,
            keys,
        )

        updates, opt_state = self.optim.update(
            grads, self.opt_state, eqx.filter(self.model, eqx.is_array)
        )

        model = eqx.apply_updates(self.model, updates)

        # 2nd train step (optional): flow matching on successful experiences from replay buffer
        if self.replay_buffer is not None:
            replay_buffer = self.replay_buffer.add(
                scene_keys=jnp.full(
                    (self.batch_size,), scene_key, dtype=scene_key.dtype
                ),
                path_candidates=path_candidates,
                rewards=rewards,
            )

            scene_keys, path_candidates, rewards = replay_buffer.sample(
                self.batch_size, key=key
            )

            grads = eqx.filter_grad(
                partial(replay_loss, scene_fn=self.scene_fn, debug=self.debug)
            )(
                model,
                scene_keys,
                path_candidates,
                rewards,
            )

            updates, opt_state = self.optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )

            model = eqx.apply_updates(model, updates)

        return (
            eqx.tree_at(
                lambda agent: (
                    agent.model,
                    agent.opt_state,
                    agent.steps_count,
                    agent.replay_buffer,
                ),
                self,
                (model, opt_state, self.steps_count + 1, replay_buffer),
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
            scene = self.scene_fn(key=scene_key)
            path_candidates = jax.vmap(
                lambda key: self.model(scene, inference=True, key=key)
            )(jr.split(key, num_path_candidates))
            return accuracy(scene, path_candidates), hit_rate(scene, path_candidates)

        num_scenes = scene_keys.shape[0]
        keys = jr.split(key, num_scenes)

        accuracies, hit_rates = jax.vmap(_evaluate)(scene_keys, keys)

        return accuracies.mean(), jnp.nanmean(hit_rates)

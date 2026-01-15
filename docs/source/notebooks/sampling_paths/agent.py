from collections.abc import Callable
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

from .metrics import reward
from .model import Model

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


def flow_matching_loss(
    model: Model,
    scene: TriangleScene,
    key: PRNGKeyArray,
    reward_fn: Callable[[Int[Array, "order"], TriangleScene], Float[Array, ""]]
    | None = None,
) -> tuple[Float[Array, ""], tuple[Int[Array, "order"], Float[Array, ""]]]:
    """
    Compute the flow matching loss for a given model and scene.
    """
    partial_path_candidate = -jnp.ones(model.order, dtype=int)
    last_object = jnp.array(-1)
    if reward_fn is None:
        reward_fn = reward
    parent_flow_key, key = jr.split(key)
    parent_flows = model.flow(
        scene,
        partial_path_candidate,
        last_object,
        key=parent_flow_key,
    )

    # Sums of flow mismatch
    flow_mismatch = jnp.array(0.0)

    for i, key in enumerate(jr.split(key, model.order)):
        edge_flow_key, action_key = jr.split(key)
        action = jr.categorical(action_key, logits=jnp.log(parent_flows))
        partial_path_candidate = partial_path_candidate.at[i].set(action)
        last_object = action

        if i == model.order - 1:
            R = reward_fn(partial_path_candidate, scene)
            edge_flows = jnp.zeros_like(parent_flows)
        else:
            R = 0.0
            edge_flows = model.flow(
                scene,
                partial_path_candidate,
                last_object,
                key=edge_flow_key,
            )

        flow_mismatch += (parent_flows[action] - edge_flows.sum() - R) ** 2

        parent_flows = edge_flows

    return flow_mismatch, (partial_path_candidate, R)


def loss(
    model: Model,
    scene: TriangleScene,
    keys: Key[Array, " batch_size"],
    reward_fn: Callable[[Int[Array, "order"], TriangleScene], Float[Array, ""]]
    | None = None,
) -> tuple[
    Float[Array, " "],
    tuple[Int[Array, "batch_size order"], Float[Array, " batch_size"]],
]:
    loss_values, aux = jax.vmap(
        flow_matching_loss, in_axes=(None, None, 0, None)
    )(model, scene, keys, reward_fn)
    return loss_values.mean(), aux


class Agent(eqx.Module):
    """
    Agent that learns to sample path candidates on triangle scenes.
    It uses flow matching to learn a flow model that estimates the flow of paths
    through the scene, and uses this flow model to sample path candidates.

    Additionally, it uses a memory to store past experiences of sampled path candidates
    and their rewards, which can be used to quickly compute averaged rewards and train the flow model
    on path experiences by imposing that the total flow matches the total number of valid paths.
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
        reward_fn: Callable[
            [Int[Array, "order"], TriangleScene], Float[Array, ""]
        ]
        | None = None,
    ) -> tuple[Self, Float[Array, " "], Float[Array, " "]]:
        """
        Train the model on one scene using the flow matching loss.
        """

        (loss_value, (path_candidates, rewards)), grads = (
            eqx.filter_value_and_grad(loss, has_aux=True)(
                self.model, scene, jr.split(key, self.batch_size), reward_fn
            )
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
            rewards.mean(),
        )

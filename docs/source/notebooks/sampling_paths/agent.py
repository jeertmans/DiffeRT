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

from .generators import random_scene
from .memory import Memory
from .metrics import reward
from .model import Model

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


class Agent(eqx.Module):
    # Static
    order: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    augmented: bool = eqx.field(static=True)

    # Trainable
    model: Model

    # Updatable
    memory: Memory
    optim: optax.GradientTransformationExtraArgs
    opt_state: optax.OptState
    steps_count: Int[Array, " "]

    def __init__(
        self,
        *,
        order: int,
        num_embeddings: int = 128,
        width_size: int = 256,
        depth: int = 3,
        batch_size: int = 64,
        optim: optax.GradientTransformationExtraArgs | None = None,
        augmented: bool = False,
        memory_size: int = 10_000,
        key: PRNGKeyArray,
    ) -> None:
        self.order = order
        self.batch_size = batch_size
        self.augmented = augmented

        assert not augmented

        self.model = Model(
            order=order,
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            key=key,
        )

        self.memory = Memory(
            order=order, memory_size=memory_size, key_dtype=key.dtype
        )
        self.optim = optax.adam(3e-5) if optim is None else optim
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.steps_count = jnp.array(0)

    @eqx.filter_jit
    def train_flow_model(
        self, *, key: PRNGKeyArray
    ) -> tuple[Self, Float[Array, " "]]:
        def loss(
            model: Model, key: PRNGKeyArray
        ) -> tuple[
            Float[Array, " "], tuple[Int[Array, " order"], Float[Array, " "]]
        ]:
            partial_path_candidate = -jnp.ones(self.order, dtype=int)
            parent_flow_key, key = jr.split(key)
            parent_flows = model.flow(
                scene,
                partial_path_candidate,
                key=parent_flow_key,
            )

            # Sums of forward and backward flows (policies)
            sum_log_P_F = 0.0
            sum_log_P_B = 0.0
            flow_mismatch = 0.0

            for i, key in enumerate(jr.split(key, self.order)):
                edge_flow_key, action_key = jr.split(key)

                logits = parent_flows
                action = jr.categorical(action_key, logits=logits)
                partial_path_candidate = partial_path_candidate.at[i].set(
                    action
                )

                sum_log_P_F += jax.nn.log_softmax(logits)[action]

                # jax.debug.print("Action = {a}", a=action)

                edge_flows = model.flow(
                    scene, partial_path_candidate, key=edge_flow_key
                )

                if i == (self.order - 1):
                    sum_edge_flows = reward(
                        partial_path_candidate.reshape(1, -1), scene
                    ).reshape(())
                else:
                    sum_edge_flows = edge_flows.sum()

                flow_mismatch += (parent_flows[action] - sum_edge_flows) ** 2

                sum_log_P_B += jnp.log(1)

                parent_flows = edge_flows

            path_candidate = partial_path_candidate
            R = reward(path_candidate.reshape(1, -1), scene).reshape(())
            log_R = jnp.log(R).clip(min=-100.0)
            log_Z = jnp.log(model.Z(scene)).clip(min=-100.0)

            tb_loss = (log_Z + sum_log_P_F - log_R - sum_log_P_B) ** 2

            return tb_loss, (path_candidate, R)

        def batch_loss(
            model: Model,
            keys: Key[Array, " batch_size"],
        ) -> tuple[
            Float[Array, " "],
            tuple[Int[Array, "batch_size order"], Float[Array, " batch_size"]],
        ]:
            tb_losses, aux = jax.vmap(loss, in_axes=(None, 0))(model, keys)
            return tb_losses.mean(), aux

        scene_key, batch_loss_key = jr.split(key)
        scene = random_scene(key=scene_key)

        (losses, (path_candidates, rewards)), grads = eqx.filter_value_and_grad(
            batch_loss, has_aux=True
        )(self.model, jr.split(batch_loss_key, self.batch_size))

        scene_keys = jnp.repeat(scene_key, self.batch_size)

        mem = self.memory.add_experiences(
            scene_keys,
            path_candidates,
            rewards,
        )

        updates, opt_state = self.optim.update(
            grads, self.opt_state, eqx.filter(self.model, eqx.is_array)
        )

        return (
            eqx.tree_at(
                lambda agent: (
                    agent.model,
                    agent.opt_state,
                    agent.memory,
                    agent.steps_count,
                ),
                self,
                (
                    eqx.apply_updates(self.model, updates),
                    opt_state,
                    mem,
                    self.steps_count + 1,
                ),
            ),
            losses.mean(),
        )

    @eqx.filter_jit
    def train_Z_model(
        self, *, key: PRNGKeyArray
    ) -> tuple[Self, Float[Array, " "]]:
        def loss(model: Model, key: PRNGKeyArray) -> Float[Array, " "]:
            scene = random_scene(key=key)
            num_valid_paths = scene.compute_paths(order=self.order).mask.sum()
            Z = model.Z(scene)
            delta = Z - num_valid_paths
            # We penalize more if the model under-estimates
            return jnp.where(delta > 0, delta, -10 * delta)

        def batch_loss(
            model: Model,
            keys: Key[Array, " batch_size"],
        ) -> Float[Array, " "]:
            return jax.vmap(loss, in_axes=(None, 0))(model, keys).mean()

        losses, grads = eqx.filter_value_and_grad(batch_loss)(
            self.model, jr.split(key, self.batch_size)
        )

        updates, opt_state = self.optim.update(
            grads, self.opt_state, eqx.filter(self.model, eqx.is_array)
        )

        return (
            eqx.tree_at(
                lambda agent: (
                    agent.model,
                    agent.opt_state,
                ),
                self,
                (
                    eqx.apply_updates(self.model, updates),
                    opt_state,
                ),
            ),
            losses.mean(),
        )

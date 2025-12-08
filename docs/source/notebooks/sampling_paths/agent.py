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
    jaxtyped,
)
from beartype import beartype as typechecker

from .generators import random_scene
from .memory import Memory
from .metrics import reward
from .model import Model

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


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
    order: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

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
        dropout_rate: float = 0.15,
        batch_size: int = 64,
        optim: optax.GradientTransformationExtraArgs | None = None,
        memory_size: int = 10_000,
        key: PRNGKeyArray,
    ) -> None:
        self.order = order
        self.batch_size = batch_size

        self.model = Model(
            order=order,
            num_embeddings=num_embeddings,
            width_size=width_size,
            depth=depth,
            dropout_rate=dropout_rate,
            key=key,
        )

        self.memory = Memory(
            order=order, memory_size=memory_size, key_dtype=key.dtype
        )
        self.optim = optax.adam(3e-5) if optim is None else optim
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_array))
        self.steps_count = jnp.array(0)

    @eqx.filter_jit
    def train_flow_matching(
        self, *, key: PRNGKeyArray
    ) -> tuple[Self, Float[Array, " "]]:
        @jaxtyped(typechecker=typechecker)
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

            # Sums of flow mismatch
            flow_mismatch = jnp.array(0.0)

            for i, key in enumerate(jr.split(key, self.order)):
                edge_flow_key, action_key = jr.split(key)
                action = jr.categorical(action_key, logits=jnp.log(parent_flows))
                partial_path_candidate = partial_path_candidate.at[i].set(
                    action
                )

                if i == self.order - 1:
                    path_candidate = partial_path_candidate
                    R = reward(path_candidate.reshape(1, -1), scene).reshape(())
                    edge_flows = jnp.zeros_like(parent_flows)
                else:
                    R = 0.0
                    edge_flows = model.flow(
                    scene, partial_path_candidate, key=edge_flow_key
                    )

                flow_mismatch += (parent_flows[action] - edge_flows.sum() - R) ** 2

                parent_flows = edge_flows


            return flow_mismatch, (path_candidate, R)

        @jaxtyped(typechecker=typechecker)
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
    def train_total_flow(
        self, *, key: PRNGKeyArray
    ) -> tuple[Self, Float[Array, " "]]:
        @jaxtyped(typechecker=typechecker)
        def loss(model: Model, key: PRNGKeyArray) -> Float[Array, " "]:
            scene = random_scene(key=key)
            num_valid_paths = scene.compute_paths(order=self.order).mask.sum()
            parent_flows = model.flow(
                scene,
                -jnp.ones(self.order, dtype=int),
                key=key,
            )
            return (num_valid_paths - parent_flows.sum())**2

        @jaxtyped(typechecker=typechecker)
        def batch_loss(
            model: Model,
            keys: Key[Array, " batch_size"],
        ) -> Float[Array, " "]:
            return jax.vmap(loss, in_axes=(None, 0))(model, keys).mean()

        # Sample scene (keys) from memory, with higher probability for scene who obtained higher rewards
        keys, _, _ = self.memory.sample_experiences(
            batch_size=self.batch_size, key=key
        )

        losses, grads = eqx.filter_value_and_grad(batch_loss)(
            self.model, keys
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

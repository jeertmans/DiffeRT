from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    Int,
    Key,
    PRNGKeyArray,
)

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


class ReplayBuffer(eqx.Module):
    """A buffer to store successful experiences and replay them during training."""

    capacity: int = eqx.field(static=True)
    """Maximum number of experiences to store in the buffer."""

    scene_keys: Key[Array, " capacity"]
    """Keys to re-generate the scenes."""
    path_candidates: Int[Array, "capacity order"]
    """Sampled path candidates."""
    rewards: Float[Array, " capacity"]
    """Rewards obtained for the sampled path candidates."""
    counter: Int[Array, " "]
    """Counter to keep track of the number of stored experiences."""

    def __init__(self, capacity: int, order: int, scene_key_dtype: jnp.dtype) -> None:
        self.capacity = capacity

        self.scene_keys = jnp.empty((capacity,), dtype=scene_key_dtype)
        self.path_candidates = -jnp.ones((capacity, order), dtype=int)
        self.rewards = jnp.empty((capacity,))

        self.counter = jnp.array(0)

    @eqx.filter_jit
    def add(
        self,
        scene_keys: Key[Array, " batch_size"],
        path_candidates: Int[Array, "batch_size trajectory_length"],
        rewards: Float[Array, " batch_size"],
    ) -> Self:
        """
        Add successful experiences to the replay buffer.

        Args:
            scene_keys: Keys to re-generate the scenes.
            path_candidates: Sampled path candidates.
            rewards: Rewards obtained for the sampled path candidates.
                A successful experience is defined as one with reward > 0.

        Returns:
            Updated buffer with new experiences added.
        """
        batch_size = scene_keys.shape[0]
        indices = (self.counter + jnp.arange(batch_size)) % self.capacity

        successful = rewards > 0.0
        num_successful = successful.sum()

        sort_idx = jnp.argsort(successful, descending=True)
        scene_keys = scene_keys[sort_idx]
        path_candidates = path_candidates[sort_idx]
        rewards = rewards[sort_idx]

        indices = jnp.where(
            jnp.arange(batch_size) < num_successful,
            indices,
            -1,
        )  # Only store successful experiences

        return eqx.tree_at(
            lambda buf: (
                buf.scene_keys,
                buf.path_candidates,
                buf.rewards,
                buf.counter,
            ),
            self,
            (
                self.scene_keys.at[indices].set(
                    scene_keys, wrap_negative_indices=False
                ),
                self.path_candidates.at[indices].set(
                    path_candidates, wrap_negative_indices=False
                ),
                self.rewards.at[indices].set(rewards, wrap_negative_indices=False),
                self.counter + num_successful,
            ),
        )

    @eqx.filter_jit
    def sample(
        self, batch_size: int, *, key: PRNGKeyArray
    ) -> tuple[
        Key[Array, " batch_size"],
        Int[Array, " batch_size order"],
        Float[Array, " batch_size"],
    ]:
        # JIT-friendly way of sampling indices up to min(counter, capacity)
        indices = jnp.arange(self.capacity)
        p = self.rewards > 0.0
        p = jnp.where(indices >= self.counter, 0.0, p)
        indices = jr.choice(key, indices, (batch_size,), replace=False, p=p)
        return (
            self.scene_keys[indices],
            self.path_candidates[indices],
            self.rewards[indices],
        )

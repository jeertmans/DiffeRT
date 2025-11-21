from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype as typechecker
from jaxtyping import (
    Array,
    Float,
    Int,
    Key,
    PRNGKeyArray,
    jaxtyped,
)

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


class Memory(eqx.Module):
    """Memory to hold 'experiences' about sampled path candidates.

    This serves two purposes:
    1. Allowing to compute averaged reward over multiple samples;
    2. Replaying past experiences to learn other goals,
       like the estimated number of valid paths on a given scene.
    """

    memory_size: int = eqx.field(static=True)

    scene_keys: Key[Array, " memory_size"]
    # We don't store scenes, but rather keys that we can use to re-generate a scene.
    path_candidates: Int[Array, "memory_size order"]
    rewards: Float[Array, " memory_size"]
    counter: Int[Array, " "]

    def __init__(self, memory_size: int, order: int, key_dtype: jnp.dtype) -> None:
        self.memory_size = memory_size

        self.scene_keys = jnp.empty((memory_size,), dtype=key_dtype)
        self.path_candidates = -jnp.ones((memory_size, order), dtype=int)
        self.rewards = jnp.empty((memory_size,))

        self.counter = jnp.array(0)

    @eqx.filter_jit
    def add_experiences(
        self,
        scene_keys: Key[Array, " batch_size"],
        path_candidates: Int[Array, "batch_size trajectory_length"],
        rewards: Float[Array, " batch_size"],
    ) -> Self:
        batch_size = scene_keys.shape[0]
        indices = (self.counter + jnp.arange(batch_size)) % self.memory_size
        return eqx.tree_at(
            lambda mem: (
                mem.scene_keys,
                mem.path_candidates,
                mem.rewards,
                mem.counter,
            ),
            self,
            (
                self.scene_keys.at[indices].set(
                    scene_keys, wrap_negative_indices=False
                ),
                self.path_candidates.at[indices].set(
                    path_candidates, wrap_negative_indices=False
                ),
                self.rewards.at[indices].set(
                    rewards, wrap_negative_indices=False
                ),
                self.counter + batch_size,
            ),
        )

    @eqx.filter_jit
    def average_reward(self) -> Float[Array, " "]:
        return jnp.mean(
            self.rewards, where=(self.path_candidates != -1).all(axis=-1)
        )

    @eqx.filter_jit
    def average_num_valid_path_candidates_per_data_key(
        self,
    ) -> Float[Array, " "]:
        keys = jr.key_data(self.scene_keys)
        traj = self.path_candidates
        rews = self.rewards
        # Sort by data key and then by trajectory
        idx = jnp.lexsort((*keys.T, *traj.T)[::-1])
        keys = keys[idx]
        traj = traj[idx]
        rews = rews[idx]
        # True each time the key changes
        key_changes = jnp.concat((
            jnp.array([True]),
            (jnp.diff(keys, axis=0) != 0).all(axis=1),
        ))
        key_changes.sum()
        # True each time the trajectory changes
        traj_changes = jnp.concat((
            jnp.array([True]),
            (jnp.diff(traj, axis=0) != 0).all(axis=1),
        ))
        traj_changes |= key_changes  # New key means new traj
        traj_changes &= rews > 0  # We only count valid path_candidates

        return traj_changes.sum() / key_changes.sum()

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def sample_experiences(
        self, batch_size: int, *, key: PRNGKeyArray
    ) -> tuple[
        Key[Array, " batch_size"],
        Int[Array, "batch_size trajectory_length"],
        Float[Array, " batch_size"],
    ]:
        # JIT-friendly way of sampling indices up to min(counter, memory_size)
        indices = jnp.arange(self.memory_size)
        p = jnp.where(self.rewards[indices] > 0, 10.0, 1.0)
        p = jnp.where(indices >= self.counter, 0.0, p)
        indices = jr.choice(key, indices, (batch_size,), replace=False, p=p)
        return (
            self.scene_keys[indices],
            self.path_candidates[indices],
            self.rewards[indices],
        )

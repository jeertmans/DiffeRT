import chex
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray

from .replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def test_add_and_sample(self, order: int, key: PRNGKeyArray) -> None:
        scene_key, sample_key = jr.split(key)
        buf = ReplayBuffer(capacity=10, order=order, scene_key_dtype=scene_key.dtype)

        # 1 - Add 4 experiences, 2 successful

        scene_keys = jr.split(scene_key, 4)
        path_candidates = jnp.arange(4 * order).reshape(4, order)
        rewards = jnp.array([0.0, 1.0, 0.0, 1.0])

        buf = buf.add(
            scene_keys=scene_keys,
            path_candidates=path_candidates,
            rewards=rewards,
        )

        assert buf.counter == 2  # Only 2 successful experiences added
        idx = jnp.argwhere(rewards > 0).flatten()
        chex.assert_trees_all_equal(buf.scene_keys[:2], scene_keys[idx])
        chex.assert_trees_all_equal(buf.path_candidates[:2], path_candidates[idx])
        chex.assert_trees_all_equal(buf.rewards[:2], 1.0)

        _, _, sampled_rewards = buf.sample(batch_size=2, key=sample_key)

        chex.assert_trees_all_equal(sampled_rewards, 1.0)

        _, _, sampled_rewards = buf.sample(batch_size=10, key=sample_key)
        with pytest.raises(AssertionError):
            # Not enough successful experiences

            chex.assert_trees_all_equal(sampled_rewards, 1.0)

        # 2 - Add 10 experiences, 2 successful

        scene_keys = jr.split(scene_key, 10)
        path_candidates = jnp.arange(10 * order).reshape(10, order)
        rewards = jnp.zeros(10).at[4:6].set(1.0)

        buf = buf.add(
            scene_keys=scene_keys,
            path_candidates=path_candidates,
            rewards=rewards,
        )

        assert buf.counter == 4  # Only 4 successful experiences
        idx = jnp.argwhere(rewards > 0).flatten()
        chex.assert_trees_all_equal(buf.scene_keys[2:4], scene_keys[idx])
        chex.assert_trees_all_equal(buf.path_candidates[2:4], path_candidates[idx])
        chex.assert_trees_all_equal(buf.rewards[2:4], 1.0)

        _, _, sampled_rewards = buf.sample(batch_size=4, key=sample_key)

        chex.assert_trees_all_equal(sampled_rewards, 1.0)

        _, _, sampled_rewards = buf.sample(batch_size=10, key=sample_key)
        with pytest.raises(AssertionError):
            # Not enough successful experiences

            chex.assert_trees_all_equal(sampled_rewards, 1.0)

        # 3 - Add 10 successful experiences

        scene_keys = jr.split(scene_key, 10)
        path_candidates = jnp.arange(10 * order).reshape(10, order)
        rewards = jnp.ones(10)

        buf = buf.add(
            scene_keys=scene_keys,
            path_candidates=path_candidates,
            rewards=rewards,
        )

        assert buf.counter == 14  # 14 successful experiences, but capacity is 10
        # Experiences are stored in a circular buffer manner
        idx = (jnp.arange(10) + 4) % buf.capacity
        chex.assert_trees_all_equal(buf.scene_keys[idx], scene_keys)
        chex.assert_trees_all_equal(buf.path_candidates[idx], path_candidates)
        chex.assert_trees_all_equal(buf.rewards, 1.0)

        _, _, sampled_rewards = buf.sample(batch_size=10, key=sample_key)

        chex.assert_trees_all_equal(sampled_rewards, 1.0)

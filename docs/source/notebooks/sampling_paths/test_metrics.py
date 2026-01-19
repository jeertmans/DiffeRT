# ruff: noqa: B020, PLR1704
import chex
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray
from pytest_subtests import SubTests

from .generators import random_scene
from .metrics import accuracy, hit_rate, reward_fn


@pytest.mark.parametrize("order", [0, 1, 2])
def test_reward_fn(order: int, key: PRNGKeyArray, subtests: SubTests) -> None:
    for i, key in enumerate(jr.split(key, 20)):
        with subtests.test(i=i):
            scene = random_scene(key=key)

            paths = scene.compute_paths(order=order)
            path_candidates = paths.objects[:, 1:-1]

            # Exhaustive candidates should give hit rate of 1.0
            got = reward_fn(path_candidates, scene)
            chex.assert_trees_all_equal(got, paths.mask.astype(float))

            # Valid candidates should get reward 1.0
            valid_candidates = path_candidates[paths.mask, :]
            got = reward_fn(valid_candidates, scene)
            chex.assert_trees_all_equal(got, 1.0)

            # Invalid candidates should get reward 0.0
            invalid_candidates = path_candidates[~paths.mask, :]
            got = reward_fn(invalid_candidates, scene)
            chex.assert_trees_all_equal(got, 0.0)


@pytest.mark.parametrize("order", [1, 2])
def test_accuracy(order: int, key: PRNGKeyArray, subtests: SubTests) -> None:
    for i, key in enumerate(jr.split(key, 20)):
        with subtests.test(i=i):
            scene_key, sample_key = jr.split(key, 2)
            scene = random_scene(key=scene_key)

            paths = scene.compute_paths(order=order)
            path_candidates = paths.objects[:, 1:-1]

            got = accuracy(scene, path_candidates)
            chex.assert_trees_all_equal(got, paths.mask.astype(float).mean())

            # Now test with only a subset of candidates
            indices = jr.randint(
                sample_key,
                (path_candidates.shape[0] // 2,),
                0,
                path_candidates.shape[0],
            )

            sampled_candidates = path_candidates[indices]
            got = accuracy(scene, sampled_candidates)
            expected = paths.mask[indices].astype(float).mean()
            chex.assert_trees_all_equal(got, expected)


@pytest.mark.parametrize("order", [1, 2])
def test_hit_rate(order: int, key: PRNGKeyArray, subtests: SubTests) -> None:
    for i, key in enumerate(jr.split(key, 20)):
        with subtests.test(i=i):
            scene_key, sample_key = jr.split(key, 2)
            scene = random_scene(key=scene_key)

            paths = scene.compute_paths(order=order)
            path_candidates = paths.objects[:, 1:-1]

            # Exhaustive candidates should give hit rate of 1.0
            got = hit_rate(scene, path_candidates)
            chex.assert_trees_all_equal(got, 1.0)

            # Now test with only a subset of candidates
            indices = jr.choice(
                sample_key,
                path_candidates.shape[0],
                (path_candidates.shape[0] // 2,),
                replace=False,
            )

            sampled_candidates = path_candidates[indices]
            got = hit_rate(scene, sampled_candidates)
            num_valid_paths = paths.mask.astype(float).sum()
            if num_valid_paths == 0:
                expected = 1.0
            else:
                expected = (
                    paths.mask[indices].astype(float).sum() / num_valid_paths
                )
            chex.assert_trees_all_equal(got, expected)

            # Duplicate candidates should have no effect on hit rate
            sampled_candidates = jnp.concat(
                (sampled_candidates, sampled_candidates), axis=0
            )
            got = hit_rate(scene, sampled_candidates)
            chex.assert_trees_all_equal(got, expected)

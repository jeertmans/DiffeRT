from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from differt.em.constants import c
from differt.em.utils import lengths_to_delays, path_delays
from tests.utils import random_inputs


@pytest.mark.parametrize(
    ("lengths", "speed", "expectation"),
    [
        ((10,), (1,), does_not_raise()),
        ((10,), (2,), pytest.raises(TypeError)),
        ((20, 10), (1,), does_not_raise()),
        ((20, 10), (10,), does_not_raise()),
        ((20, 10), (20,), pytest.raises(TypeError)),
        ((10, 4), (10, 5), pytest.raises(TypeError)),
    ],
)
@random_inputs("lengths", "speed")
def test_lengths_to__delays_random_inputs(
    lengths: Array,
    speed: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = lengths_to_delays(lengths, speed=speed)
        expected = lengths / speed

        chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    ("paths", "expectation"),
    [
        ((10, 3), does_not_raise()),
        ((20, 10, 3), does_not_raise()),
        ((10, 4), pytest.raises(TypeError)),
        ((1, 3), does_not_raise()),
        ((0, 3), does_not_raise()),
    ],
)
@random_inputs("paths")
def test_path_delays_random_inputs(
    paths: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = path_delays(paths)
        expected = (
            jnp.sum(jnp.linalg.norm(jnp.diff(paths, axis=-2), axis=-1), axis=-1) / c
        )

        chex.assert_trees_all_close(got, expected)

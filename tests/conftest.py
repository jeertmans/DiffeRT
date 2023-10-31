from collections.abc import Iterator

import jax
import pytest


@pytest.fixture
def seed() -> Iterator[int]:
    return 1234


@pytest.fixture
def key(seed) -> Iterator[jax.random.PRNGKey]:
    return jax.random.key(seed)

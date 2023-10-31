import pytest

import jax
from typing import Iterator

@pytest.fixture
def seed() -> Iterator[int]:
    return 1234


@pytest.fixture
def key(seed) -> Iterator[jax.random.PRNGKey]:
    return jax.random.key(seed)


import jax
import pytest
from jaxtyping import PRNGKeyArray


@pytest.fixture
def seed() -> int:
    return 1234


@pytest.fixture
def key(seed: int) -> PRNGKeyArray:
    return jax.random.key(seed)

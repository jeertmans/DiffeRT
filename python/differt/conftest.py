# pragme: no cover
from typing import Any

import jax
import numpy
import pytest


@pytest.fixture(autouse=True)
def add_doctest_modules(doctest_namespace: dict[str, Any]) -> None:
    doctest_namespace["jax"] = jax
    doctest_namespace["jnp"] = jax.numpy
    doctest_namespace["np"] = numpy

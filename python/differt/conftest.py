from typing import Any

import jax
import numpy
import pytest


@pytest.fixture(autouse=True)
def add_doctest_modules(doctest_namespace: dict[str, Any]) -> None:
    doctest_namespace["jax"] = jax
    doctest_namespace["jnp"] = jax.numpy
    doctest_namespace["np"] = numpy


@pytest.fixture(autouse=True)
def set_printoptions() -> None:
    # We need to do that because floats seem to vary accross OSes:
    # https://github.com/numpy/numpy/issues/21209.
    jax.numpy.set_printoptions(precision=7, suppress=True)

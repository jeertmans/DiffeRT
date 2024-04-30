from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest


@pytest.fixture(autouse=True)
def add_doctest_modules(doctest_namespace: dict[str, Any]) -> None:
    doctest_namespace["go"] = go
    doctest_namespace["jax"] = jax
    doctest_namespace["jnp"] = jnp
    doctest_namespace["plt"] = plt
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)
def set_printoptions() -> None:
    # We need to do that because floats seem to vary across OSes:
    # https://github.com/numpy/numpy/issues/21209.
    jax.numpy.set_printoptions(precision=7, suppress=True)

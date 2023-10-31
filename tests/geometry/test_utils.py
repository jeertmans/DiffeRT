import pytest
from typing import Optional
import chex

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from differt.geometry.utils import pairwise_cross

from tests.utils import asjaxarray


@pytest.mark.parametrize(
    ("u,v,raises"),
    [
        (np.random.rand(10, 3), np.random.rand(10, 3), None),
        (np.random.rand(10, 3), np.random.rand(20, 3), None),
        (np.random.rand(10, 4), np.random.rand(20, 4), TypeError),
    ],
)
@asjaxarray("u", "v")
def test_pairwise_cross(u: Array, v: Array, raises: Optional[Exception]) -> None:
    if raises:
        with pytest.raises(raises):
            _ = pairwise_cross(u, v)
    else:
        got = pairwise_cross(u, v)

        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                expected = jnp.cross(u[i, :], v[j, :])
                chex.assert_trees_all_equal(got[i, j], expected)

import chex
import jax
import jax.experimental
import jax.numpy as jnp
import pytest
from jaxtyping import DTypeLike

from differt.em._interaction_type import InteractionType


class TestInteractionType:
    @pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64])
    def test_array(self, dtype: DTypeLike) -> None:
        with jax.experimental.enable_x64(dtype == jnp.int64):
            arr = jnp.array(list(InteractionType), dtype=dtype)
            assert arr.dtype == dtype

            for i_type in InteractionType:
                assert jnp.where(arr == i_type, 1, 0).sum() == 1

            arr = jnp.array([0, 1, 2, *list(InteractionType)], dtype=dtype)
            assert arr.dtype == dtype

    def test_values(self) -> None:
        # This is important to avoid breaking changes
        assert InteractionType.REFLECTION == 0
        assert InteractionType.DIFFRACTION == 1
        assert InteractionType.SCATTERING == 2

    def test_where(self) -> None:
        interaction_types = jnp.array([0, 1, 2, 0, 1, 2])
        x = jnp.array([1, 2, 3, 1, 2, 3])

        chex.assert_trees_all_equal(
            jnp.where(interaction_types == InteractionType.REFLECTION, x, 0),
            jnp.array([1, 0, 0, 1, 0, 0]),
        )
        chex.assert_trees_all_equal(
            jnp.where(interaction_types == InteractionType.DIFFRACTION, x, 0),
            jnp.array([0, 2, 0, 0, 2, 0]),
        )
        chex.assert_trees_all_equal(
            jnp.where(interaction_types == InteractionType.SCATTERING, x, 0),
            jnp.array([0, 0, 3, 0, 0, 3]),
        )

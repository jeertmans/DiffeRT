# ruff: noqa: FURB152
from typing import ClassVar

import chex
import jax
import jax.experimental
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from differt.em._material import Material, materials


class TestITU:
    materials: ClassVar[dict[str, Material]] = {
        name: material for name, material in materials.items() if name.startswith("itu")
    }

    def test_constructor(self) -> None:
        with pytest.raises(
            ValueError,
            match="Only one frequency range can be used if 'None' is passed, as it will match any frequency",
        ):
            _ = Material.from_itu_properties(
                "test", (0, 0, 0, 0, None), (0, 0, 0, 0, None)
            )

    def test_num_materials(self) -> None:
        assert len(self.materials) == 15

    def test_vacuum(self, key: PRNGKeyArray) -> None:
        mat = self.materials["itu_vacuum"]

        rel_perm, cond = mat.properties(1e9)

        chex.assert_trees_all_equal_shapes_and_dtypes(jnp.array(1e9), rel_perm, cond)
        chex.assert_trees_all_close(rel_perm, 1.0)
        chex.assert_trees_all_close(cond, 0.0)

        f = jax.random.randint(key, (10000, 30), 0, 100e9).astype(float)

        rel_perm, cond = mat.relative_permittivity(f), mat.conductivity(f)

        chex.assert_trees_all_equal_shapes_and_dtypes(f, rel_perm, cond)
        chex.assert_trees_all_close(rel_perm, 1.0)
        chex.assert_trees_all_close(cond, 0.0)

    def test_concrete(self) -> None:
        mat = self.materials["itu_concrete"]

        f = jnp.array([0.1e9, 1e9, 10e9, 100e9, 1000e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 5.24, 5.24, 5.24, -1.0])
        expected_cond = jnp.array([-1.0, 0.0462, 0.279796, 1.694501, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_glass(self) -> None:
        mat = self.materials["itu_glass"]

        f = jnp.array([0.01e9, 0.1e9, 10e9, 100e9, 150e9, 220e9, 350e9, 450e9, 500e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([
            -1.0,
            6.31,
            6.31,
            6.31,
            -1.0,
            5.79,
            5.79,
            5.79,
            -1.0,
        ])
        expected_cond = jnp.array([
            -1.0,
            1.647792e-04,
            7.865069e-02,
            1.718314,
            -1.0,
            3.060531,
            6.608833,
            1.002504e01,
            -1.0,
        ])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_ceiling_board(self) -> None:
        mat = self.materials["itu_ceiling_board"]

        f = jnp.array([0.1e9, 1e9, 10e9, 100e9, 150e9, 220e9, 350e9, 450e9, 500e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([
            -1.0,
            1.48,
            1.48,
            1.48,
            -1.0,
            1.52,
            1.52,
            1.52,
            -1.0,
        ])
        expected_cond = jnp.array([
            -1.0,
            1.100000e-03,
            1.307353e-02,
            1.553792e-01,
            -1.0,
            7.460210e-01,
            1.202940,
            1.557951,
            -1.0,
        ])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_plywood(self) -> None:
        mat = self.materials["itu_plywood"]

        f = jnp.array([0.1e9, 1e9, 10e9, 40e9, 100e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 2.71, 2.71, 2.71, -1.0])
        expected_cond = jnp.array([-1.0, 0.33, 0.33, 0.33, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_metal(self) -> None:
        mat = self.materials["itu_metal"]

        f = jnp.array([0.1e9, 1e9, 10e9, 100e9, 1000e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 1.0, 1.0, 1.0, -1.0])
        expected_cond = jnp.array([-1.0, 1e7, 1e7, 1e7, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

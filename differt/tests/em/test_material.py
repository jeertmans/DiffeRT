# ruff:file-ignore[math-constant]
from typing import ClassVar

import chex
import jax
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
                "test", (0.0, 0.0, 0.0, 0.0, None), (0.0, 0.0, 0.0, 0.0, None)
            )

    def test_num_materials(self) -> None:
        assert len(self.materials) == 19

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

    def test_concrete_scalar(self) -> None:
        mat = self.materials["itu_concrete"]

        for f, expected_rel_perm, expected_cond in zip(
            [0.1e9, 1e9, 10e9, 100e9, 1000e9],
            [-1.0, 5.24, 5.24, 5.24, -1.0],
            [-1.0, 0.0462, 0.279796, 1.694501, -1.0],
            strict=False,
        ):
            got_rel_perm, got_cond = mat.properties(f)
            chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
            chex.assert_trees_all_close(got_cond, expected_cond)

    def test_glass(self) -> None:
        mat = self.materials["itu_glass"]

        f = jnp.array([0.01e9, 0.1e9, 10e9, 100e9, 150e9, 220e9, 350e9, 450e9, 500e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([
            -1.0,
            6.27,
            6.27,
            6.27,
            6.70,
            6.70,
            6.70,
            6.01,
            -1.0,
        ])
        expected_cond = jnp.array([
            -1.0,
            0.0002760377246886492,
            0.06698359549045563,
            1.0434422492980957,
            1.335839033126831,
            2.0750820636749268,
            3.539381980895996,
            5.6384806632995605,
            -1.0,
        ])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_ceiling_board(self) -> None:
        mat = self.materials["itu_ceiling_board"]

        f = jnp.array([0.1e9, 1e9, 10e9, 100e9, 150e9, 220e9, 350e9, 400e9, 500e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([
            -1.0,
            1.48,
            1.48,
            1.48,
            1.58,
            1.58,
            1.58,
            1.58,
            -1.0,
        ])
        expected_cond = jnp.array([
            -1.0,
            0.0011,
            0.01476361,
            0.19814934,
            0.29822615,
            0.4492834,
            0.7383817,
            0.8517896,
            -1.0,
        ])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_plywood(self) -> None:
        mat = self.materials["itu_plywood"]

        f = jnp.array([0.1e9, 1e9, 10e9, 40e9, 100e9, 400e9, 500e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 2.71, 2.71, 2.71, 2.17, 2.17, -1.0])
        expected_cond = jnp.array([
            -1.0,
            0.33,
            0.33,
            0.33,
            0.7750691175460815,
            3.2998414039611816,
            -1.0,
        ])
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

    def test_wet_ground(self) -> None:
        mat = self.materials["itu_wet_ground"]

        f = jnp.array([0.1e9, 1e9, 10e9, 100e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 30.0, 11.943215, -1.0])
        expected_cond = jnp.array([-1.0, 0.15, 2.992893, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_clear_acrylic(self) -> None:
        mat = self.materials["itu_clear_acrylic"]

        f = jnp.array([0.1e9, 1e9, 10e9, 40e9, 100e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 2.57, 2.57, 2.57, -1.0])
        expected_cond = jnp.array([-1.0, 0.0049, 0.05627248, 0.24464695, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_vinyl_tile(self) -> None:
        mat = self.materials["itu_vinyl_tile"]

        f = jnp.array([0.1e9, 1e9, 40e9, 100e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 3.62, 3.62, -1.0])
        expected_cond = jnp.array([-1.0, 0.0051, 0.11397906, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_carpet_tile(self) -> None:
        mat = self.materials["itu_carpet_tile"]

        f = jnp.array([0.1e9, 1e9, 40e9, 100e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 2.08, 2.08, -1.0])
        expected_cond = jnp.array([-1.0, 0.0009, 0.018532401, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_asphalt_concrete(self) -> None:
        mat = self.materials["itu_asphalt_concrete"]

        f = jnp.array([0.1e9, 1e9, 40e9, 100e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 4.83, 4.83, -1.0])
        expected_cond = jnp.array([-1.0, 0.0108, 1.8678477, -1.0])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)


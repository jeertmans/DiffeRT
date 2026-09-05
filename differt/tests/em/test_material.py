# ruff:file-ignore[math-constant]
from pathlib import Path
from typing import Any, ClassVar

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from differt.em import Material, MaterialsDict, materials
from differt.em._material import _populate_materials
from differt.geometry import Scene
from differt_core.geometry import SionnaScene


class TestITU:
    materials: ClassVar[MaterialsDict] = materials

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

        # 220 and 350 GHz fall in both the (100, 400) and (220, 450) GHz ranges;
        # the narrower (220, 450) range takes priority.
        expected_rel_perm = jnp.array([
            -1.0,
            6.31,
            6.31,
            6.31,
            6.5767,
            5.79,
            5.79,
            5.79,
            -1.0,
        ])
        expected_cond = jnp.array([
            -1.0,
            0.00016477919,
            0.078650691,
            1.71831405,
            1.89401102,
            3.06053066,
            6.60883188,
            10.0250435,
            -1.0,
        ])
        chex.assert_trees_all_close(got_rel_perm, expected_rel_perm)
        chex.assert_trees_all_close(got_cond, expected_cond)

    def test_ceiling_board(self) -> None:
        mat = self.materials["itu_ceiling_board"]

        f = jnp.array([0.1e9, 1e9, 10e9, 100e9, 150e9, 220e9, 350e9, 400e9, 500e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        # 220 and 350 GHz fall in both the (100, 400) and (220, 450) GHz ranges;
        # the narrower (220, 450) range takes priority.
        expected_rel_perm = jnp.array([
            -1.0,
            1.48,
            1.48,
            1.48,
            1.2567,
            1.52,
            1.52,
            1.52,
            -1.0,
        ])
        expected_cond = jnp.array([
            -1.0,
            0.0011,
            0.013073525,
            0.15537915,
            0.18966185,
            0.74602091,
            1.2029403,
            1.3801230,
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

        f = jnp.array([100e9, 110e9, 200e9, 330e9, 400e9])

        got_rel_perm, got_cond = mat.relative_permittivity(f), mat.conductivity(f)

        expected_rel_perm = jnp.array([-1.0, 2.58, 2.58, 2.58, -1.0])
        expected_cond = jnp.array([-1.0, 0.23615505, 0.6341937, 1.4507495, -1.0])
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


def _dummy_properties(_: Any) -> tuple[float, float]:
    return (1.0, 0.0)


class TestMaterialsDict:
    def test_init_and_len(self) -> None:
        d = MaterialsDict(materials)
        assert len(d) == 19

        mat = Material(
            name="Custom",
            properties=_dummy_properties,
            aliases=("custom_alias",),
        )
        d_list = MaterialsDict([mat])
        assert len(d_list) == 1
        assert "Custom" in d_list
        assert "custom_alias" in d_list

    def test_getitem_and_contains(self) -> None:
        d = MaterialsDict(materials)
        concrete = d["Concrete"]
        assert d["itu_concrete"] is concrete
        assert "Concrete" in d
        assert "itu_concrete" in d
        assert "unknown" not in d
        assert 123 not in d  # type: ignore[operator]

        with pytest.raises(KeyError):
            _ = d["unknown"]

    def test_get(self) -> None:
        d = MaterialsDict(materials)
        concrete = d.get("Concrete")
        assert d.get("itu_concrete") is concrete
        assert d.get("unknown") is None
        assert d.get("unknown", "default") == "default"

    def test_setitem_and_aliases(self) -> None:
        d = MaterialsDict()
        mat = Material(
            name="Wood", properties=_dummy_properties, aliases=("itu_wood", "timber")
        )

        # Setting via primary name
        d["Wood"] = mat
        assert len(d) == 1
        assert d["Wood"] is mat
        assert d["itu_wood"] is mat
        assert d["timber"] is mat

        # Updating existing material via alias
        new_mat = Material(
            name="Wood", properties=_dummy_properties, aliases=("itu_wood", "timber")
        )
        d["itu_wood"] = new_mat
        assert len(d) == 1
        assert d["Wood"] is new_mat

        # Setting new material via alias directly
        mat2 = Material(
            name="Brick", properties=_dummy_properties, aliases=("itu_brick",)
        )
        d["itu_brick"] = mat2
        assert len(d) == 2
        assert d["Brick"] is mat2
        assert d["itu_brick"] is mat2

    @pytest.mark.require_no_typechecker
    def test_setitem_non_material_value_new_key(self) -> None:
        # 'value' is typed as 'Material', so this bypasses the alias-registration
        # logic entirely and falls back to plain 'dict.__setitem__' -- only
        # reachable with runtime type checking disabled.
        d = MaterialsDict()
        d["raw_key"] = "not-a-material"  # type: ignore[assignment]
        assert dict.get(d, "raw_key") == "not-a-material"

    def test_delitem(self) -> None:
        d = MaterialsDict(materials)
        assert "Concrete" in d
        assert "itu_concrete" in d

        del d["itu_concrete"]
        assert "Concrete" not in d
        assert "itu_concrete" not in d
        assert len(d) == 18

        with pytest.raises(KeyError):
            del d["itu_concrete"]

    def test_pop(self) -> None:
        d = MaterialsDict(materials)
        concrete = d.pop("itu_concrete")
        assert concrete.name == "Concrete"
        assert "Concrete" not in d
        assert d.pop("itu_concrete", None) is None

        with pytest.raises(KeyError):
            d.pop("itu_concrete")

    def test_setdefault(self) -> None:
        d = MaterialsDict()
        mat = Material(
            name="Metal", properties=_dummy_properties, aliases=("itu_metal",)
        )

        res = d.setdefault("itu_metal", mat)
        assert res is mat
        assert d["Metal"] is mat

        res2 = d.setdefault("Metal", mat)
        assert res2 is mat

    def test_update_errors(self) -> None:
        d = MaterialsDict()
        with pytest.raises(TypeError, match="positional argument"):
            d.update(1, 2)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="not iterable"):
            d.update(123)  # type: ignore[arg-type]

    def test_update_with_kwargs(self) -> None:
        d = MaterialsDict()
        mat = Material(
            name="Glass", properties=_dummy_properties, aliases=("itu_glass",)
        )
        d.update(Glass=mat)
        assert len(d) == 1
        assert d["Glass"] is mat
        assert d["itu_glass"] is mat

    def test_repr(self) -> None:
        mat = Material(
            name="Test", properties=_dummy_properties, aliases=("test_alias",)
        )
        d = MaterialsDict([mat])
        assert repr(d) == "{'Test': " + repr(mat) + "}"

    def test_hash(self) -> None:
        # 'MaterialsDict' must be hashable so it can be used as (static)
        # 'GeometricFieldSolver.radio_materials' inside JIT-compiled code.
        d1 = MaterialsDict(materials)
        d2 = MaterialsDict(materials)
        assert d1 is not d2
        assert hash(d1) == hash(d2)

        d3 = MaterialsDict({k: v for k, v in materials.items() if k != "Concrete"})
        assert hash(d1) != hash(d3)


_TRIANGLE_OBJ = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
"""


def _write_scene(tmp_path: Path, bsdfs: str, shapes: str) -> Path:
    (tmp_path / "mesh.obj").write_text(_TRIANGLE_OBJ)
    scene_file = tmp_path / "scene.xml"
    scene_file.write_text(f"<scene version='2.1.0'>{bsdfs}{shapes}</scene>")
    return scene_file


def _shape(shape_id: str, material_id: str) -> str:
    return f"""
    <shape type="obj" id="{shape_id}">
        <string name="filename" value="mesh.obj"/>
        <ref id="{material_id}"/>
    </shape>
    """


class TestPopulateMaterials:
    def test_uniform_materials_share_generic_name(self, tmp_path: Path) -> None:
        bsdfs = """
        <bsdf type="itu-radio-material" id="window">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.01"/>
        </bsdf>
        """
        scene_file = _write_scene(tmp_path, bsdfs, _shape("shape-0", "window"))
        sionna_scene = SionnaScene.load_xml(scene_file)

        radio_materials: MaterialsDict = MaterialsDict(materials)
        _populate_materials(sionna_scene.materials.values(), radio_materials)

        assert "itu_glass" in radio_materials
        assert radio_materials["itu_glass"].thickness == pytest.approx(0.01)
        assert radio_materials["itu_glass"].name == "Glass"

    def test_non_uniform_materials_keyed_by_id(self, tmp_path: Path) -> None:
        bsdfs = """
        <bsdf type="itu-radio-material" id="window1">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.01"/>
        </bsdf>
        <bsdf type="itu-radio-material" id="window2">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.05"/>
        </bsdf>
        """
        shapes = _shape("shape-0", "window1") + _shape("shape-1", "window2")
        scene_file = _write_scene(tmp_path, bsdfs, shapes)
        sionna_scene = SionnaScene.load_xml(scene_file)

        radio_materials: MaterialsDict = MaterialsDict(materials)
        _populate_materials(sionna_scene.materials.values(), radio_materials)

        assert {"window1", "window2"} <= set(radio_materials)
        assert radio_materials["window1"].thickness == pytest.approx(0.01)
        assert radio_materials["window2"].thickness == pytest.approx(0.05)

    def test_non_itu_materials_are_skipped(self, tmp_path: Path) -> None:
        bsdfs = """
        <bsdf type="diffuse" id="paint">
            <rgb name="rgb" value="0.5 0.5 0.5"/>
        </bsdf>
        """
        scene_file = _write_scene(tmp_path, bsdfs, _shape("shape-0", "paint"))
        sionna_scene = SionnaScene.load_xml(scene_file)

        radio_materials: MaterialsDict = MaterialsDict(materials)
        before = dict(radio_materials)
        _populate_materials(sionna_scene.materials.values(), radio_materials)

        assert dict(radio_materials) == before

    def test_conflicting_thickness_raises(self, tmp_path: Path) -> None:
        bsdfs = """
        <bsdf type="itu-radio-material" id="window">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.01"/>
        </bsdf>
        """
        scene_file = _write_scene(tmp_path, bsdfs, _shape("shape-0", "window"))
        sionna_scene = SionnaScene.load_xml(scene_file)

        radio_materials: MaterialsDict = MaterialsDict(materials)
        _populate_materials(sionna_scene.materials.values(), radio_materials)

        bsdfs = """
        <bsdf type="itu-radio-material" id="window">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.05"/>
        </bsdf>
        """
        scene_file = _write_scene(tmp_path, bsdfs, _shape("shape-0", "window"))
        sionna_scene = SionnaScene.load_xml(scene_file)

        with pytest.raises(ValueError, match="already present"):
            _populate_materials(sionna_scene.materials.values(), radio_materials)

    def test_scene_load_xml_populates_materials(self, tmp_path: Path) -> None:
        bsdfs = """
        <bsdf type="itu-radio-material" id="window">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.01"/>
        </bsdf>
        """
        scene_file = _write_scene(tmp_path, bsdfs, _shape("shape-0", "window"))
        radio_materials: MaterialsDict = MaterialsDict(materials)
        scene = Scene.load_xml(scene_file, materials=radio_materials)

        assert scene.mesh.material_names == ("itu_glass",)
        assert radio_materials["itu_glass"].thickness == pytest.approx(0.01)

        # Populating 'materials' must not break JIT-compiled 'Scene' methods.
        _ = scene.scale(2.0)

    def test_scene_load_xml_defaults_to_global_materials(self, tmp_path: Path) -> None:
        bsdfs = """
        <bsdf type="itu-radio-material" id="window">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.01"/>
        </bsdf>
        """
        scene_file = _write_scene(tmp_path, bsdfs, _shape("shape-0", "window"))
        original_glass = materials["Glass"]
        try:
            _ = Scene.load_xml(scene_file)

            assert materials["itu_glass"].thickness == pytest.approx(0.01)
        finally:
            materials["Glass"] = original_glass

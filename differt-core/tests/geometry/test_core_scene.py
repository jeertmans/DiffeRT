from pathlib import Path

from differt_core.geometry import Scene

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


class TestScene:
    def test_append_preserves_first_shape_material(self, tmp_path: Path) -> None:
        # Regression test: 'Mesh.append' used to silently drop the material
        # name of the very first appended shape, mislabeling its faces with
        # whatever material the *second* shape happened to have.
        bsdfs = """
        <bsdf type="itu-radio-material" id="glass">
            <string name="type" value="glass"/>
        </bsdf>
        <bsdf type="itu-radio-material" id="wood">
            <string name="type" value="wood"/>
        </bsdf>
        """
        shapes = _shape("shape-0", "glass") + _shape("shape-1", "wood")
        scene = Scene.load_xml(_write_scene(tmp_path, bsdfs, shapes))

        assert scene.mesh.material_names == ["itu_glass", "itu_wood"]

        object_bounds = scene.mesh.object_bounds
        face_materials = scene.mesh.face_materials
        assert object_bounds is not None
        assert face_materials is not None

        first_start, _ = object_bounds[0]
        second_start, _ = object_bounds[1]
        assert scene.mesh.material_names[face_materials[first_start]] == "itu_glass"
        assert scene.mesh.material_names[face_materials[second_start]] == "itu_wood"

    def test_materials_keyed_by_id_only_when_thickness_differs(
        self, tmp_path: Path
    ) -> None:
        # Two shapes of the same ITU type but with different 'thickness'
        # overrides must remain distinguishable (kept under their own XML
        # id), while two shapes of the same type that agree (or that don't
        # specify a thickness at all) must still share the generic,
        # ITU-type-derived name.
        bsdfs = """
        <bsdf type="itu-radio-material" id="window1">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.01"/>
        </bsdf>
        <bsdf type="itu-radio-material" id="window2">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.05"/>
        </bsdf>
        <bsdf type="itu-radio-material" id="wall1">
            <string name="type" value="concrete"/>
        </bsdf>
        <bsdf type="itu-radio-material" id="wall2">
            <string name="type" value="concrete"/>
        </bsdf>
        """
        shapes = (
            _shape("shape-0", "window1")
            + _shape("shape-1", "window2")
            + _shape("shape-2", "wall1")
            + _shape("shape-3", "wall2")
        )
        scene = Scene.load_xml(_write_scene(tmp_path, bsdfs, shapes))

        assert scene.mesh.material_names == [
            "window1",
            "window2",
            "itu_concrete",
        ]

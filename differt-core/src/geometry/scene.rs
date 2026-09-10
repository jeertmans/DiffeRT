use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use super::sionna::SionnaScene;
use crate::geometry::mesh::Mesh;

/// Return the set of material names for which at least two materials
/// (i.e., two distinct XML ids) share that name but disagree on
/// `thickness`, meaning that name alone is not enough to tell them apart.
fn non_uniform_material_names(sionna: &SionnaScene) -> HashSet<String> {
    let mut seen: HashMap<&str, Option<f32>> = HashMap::new();
    let mut non_uniform = HashSet::new();

    for mat in sionna.materials.values() {
        match seen.get(mat.name.as_str()) {
            Some(thickness) if *thickness != mat.thickness => {
                non_uniform.insert(mat.name.clone());
            },
            Some(_) => {},
            None => {
                seen.insert(&mat.name, mat.thickness);
            },
        }
    }

    non_uniform
}

/// A scene that contains one mesh, usually being the results of multiple call to :meth:`Mesh.append<differt_core.geometry.Mesh.append>`.
///
/// This class is only useful to provide a fast constructor for scenes
/// created using the Sionna file format.
#[derive(Clone)]
#[pyclass(subclass)]
struct Scene {
    /// differt_core.geometry.Mesh: The scene mesh.
    #[pyo3(get)]
    mesh: Mesh,
}

#[pymethods]
impl Scene {
    /// Load a scene from a Sionna-compatible XML file.
    ///
    /// Args:
    ///     file (str | os.PathLike[str]): The path to the XML file.
    ///
    /// Returns:
    ///     Scene: The corresponding scene.
    #[classmethod]
    #[pyo3(name = "load_xml")]
    fn py_load_xml(_cls: &Bound<'_, PyType>, file: PathBuf) -> PyResult<Self> {
        Self::load_xml(&file)
    }
}
impl Scene {
    fn load_xml(file: &Path) -> PyResult<Self> {
        let sionna = SionnaScene::load_xml(file)?;

        let folder = file.parent().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Could not determine parent folder of file: {file:#?}",
            ))
        })?;

        let non_uniform_names = non_uniform_material_names(&sionna);

        let mut mesh = Mesh::default();

        for (_, shape) in sionna.shapes.into_iter() {
            let mesh_file_path = folder.join(shape.file);
            let mut other_mesh = match shape.r#type.as_str() {
                "obj" => Mesh::load_obj(&mesh_file_path)?,
                "ply" => Mesh::load_ply(&mesh_file_path)?,
                ty => {
                    log::warn!("Unsupported shape type {ty}, skipping.");
                    continue;
                },
            };

            let material = sionna.materials.get(&shape.material_id);

            let color = material.map(|mat| mat.color);

            // Materials whose name is shared by another, differently
            // configured material (currently, only 'thickness' can differ)
            // are kept under their unique XML id, so the two remain
            // distinguishable; every other material still shares the
            // generic, ITU-type-derived name, matching the previous
            // behavior (and keeping it resolvable against the built-in ITU
            // materials database, e.g. via `differt.em.materials_from_sionna`).
            // A material with no 'thickness' override of its own is never
            // renamed, even if its name collides with a differently
            // configured one: 'differt.em._material._populate_materials'
            // only ever creates a distinct, id-keyed entry for a material
            // that actually carries a 'thickness' (see its own leading
            // 'if mat.thickness is None: continue'), so keying a
            // thickness-less shape by id here would leave it without any
            // matching entry in the Python-side materials mapping.
            let material_name = material.map(|mat| {
                if mat.thickness.is_some() && non_uniform_names.contains(&mat.name) {
                    mat.id.clone()
                } else {
                    mat.name.clone()
                }
            });

            other_mesh.set_face_color(color.as_ref());
            other_mesh.set_face_material(material_name);

            mesh.append(&mut other_mesh);
        }
        Ok(Self { mesh })
    }
}

#[cfg(not(tarpaulin_include))]
#[pymodule(gil_used = false)]
pub(crate) fn scene(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scene>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use indexmap::IndexMap;
    use pyo3::Python;
    use tempfile::tempdir;

    use super::*;
    use crate::geometry::sionna::Material;

    const TRIANGLE_OBJ: &str = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n";

    fn shape_xml(id: &str) -> String {
        format!(
            r#"
            <shape type="obj" id="shape-{id}">
                <string name="filename" value="mesh.obj"/>
                <ref id="{id}"/>
            </shape>
            "#
        )
    }

    #[test]
    fn non_uniform_material_names_flags_same_name_different_thickness() {
        let mut materials = IndexMap::new();
        materials.insert(
            "window1".to_string(),
            Material {
                name: "itu_glass".to_string(),
                id: "window1".to_string(),
                color: [0.168, 0.139, 0.509],
                thickness: Some(0.01),
            },
        );
        materials.insert(
            "window2".to_string(),
            Material {
                name: "itu_glass".to_string(),
                id: "window2".to_string(),
                color: [0.168, 0.139, 0.509],
                thickness: Some(0.05),
            },
        );
        materials.insert(
            "wall".to_string(),
            Material {
                name: "itu_concrete".to_string(),
                id: "wall".to_string(),
                color: [0.539, 0.539, 0.539],
                thickness: None,
            },
        );
        materials.insert(
            "wall2".to_string(),
            Material {
                name: "itu_concrete".to_string(),
                id: "wall2".to_string(),
                color: [0.539, 0.539, 0.539],
                thickness: None,
            },
        );

        let sionna = SionnaScene {
            materials,
            shapes: IndexMap::new(),
        };

        let non_uniform = non_uniform_material_names(&sionna);

        // Both "itu_glass" materials share a name but disagree on
        // thickness, so the name alone cannot distinguish them.
        assert!(non_uniform.contains("itu_glass"));
        // Both "itu_concrete" materials agree (neither has a thickness
        // override), so their shared name remains sufficient.
        assert!(!non_uniform.contains("itu_concrete"));
    }

    #[test]
    fn load_xml_keeps_ids_distinguishable_when_material_names_collide() {
        let dir = tempdir().expect("failed to create temporary directory");

        fs::write(dir.path().join("mesh.obj"), TRIANGLE_OBJ).expect("failed to write obj file");

        let xml = format!(
            r#"<scene version="2.1.0">
                <bsdf type="itu-radio-material" id="window1">
                    <string name="type" value="glass"/>
                    <float name="thickness" value="0.01"/>
                </bsdf>
                <bsdf type="itu-radio-material" id="window2">
                    <string name="type" value="glass"/>
                    <float name="thickness" value="0.05"/>
                </bsdf>
                {}{}
            </scene>"#,
            shape_xml("window1"),
            shape_xml("window2"),
        );

        let scene_file = dir.path().join("scene.xml");
        fs::write(&scene_file, xml).expect("failed to write scene file");

        let material_names: Vec<String> = Python::with_gil(|py| {
            let scene = Scene::load_xml(&scene_file).expect("scene should load");
            let py_scene =
                Bound::new(py, scene).expect("failed to wrap the scene in a Python object");
            let py_mesh = py_scene
                .getattr("mesh")
                .expect("scene should have a `mesh` attribute");
            py_mesh
                .getattr("material_names")
                .expect("mesh should have a `material_names` attribute")
                .extract()
                .expect("`material_names` should be extractable as Vec<String>")
        });

        // Since the two "glass" materials disagree on thickness, they must
        // remain distinguishable under their own XML ids, rather than both
        // collapsing to the generic "itu_glass" name (which would make them
        // indistinguishable from one another).
        assert_eq!(
            material_names,
            vec!["window1".to_string(), "window2".to_string()]
        );
    }

    #[test]
    fn load_xml_keeps_generic_name_when_no_collision() {
        let dir = tempdir().expect("failed to create temporary directory");

        fs::write(dir.path().join("mesh.obj"), TRIANGLE_OBJ).expect("failed to write obj file");

        let xml = format!(
            r#"<scene version="2.1.0">
                <bsdf type="itu-radio-material" id="window">
                    <string name="type" value="glass"/>
                    <float name="thickness" value="0.01"/>
                </bsdf>
                <bsdf type="itu-radio-material" id="wall">
                    <string name="type" value="concrete"/>
                </bsdf>
                {}{}
            </scene>"#,
            shape_xml("window"),
            shape_xml("wall"),
        );

        let scene_file = dir.path().join("scene.xml");
        fs::write(&scene_file, xml).expect("failed to write scene file");

        let material_names: Vec<String> = Python::with_gil(|py| {
            let scene = Scene::load_xml(&scene_file).expect("scene should load");
            let py_scene =
                Bound::new(py, scene).expect("failed to wrap the scene in a Python object");
            let py_mesh = py_scene
                .getattr("mesh")
                .expect("scene should have a `mesh` attribute");
            py_mesh
                .getattr("material_names")
                .expect("mesh should have a `material_names` attribute")
                .extract()
                .expect("`material_names` should be extractable as Vec<String>")
        });

        // Neither material name collides with another, so both keep their
        // generic, ITU-type-derived name (resolvable against the built-in
        // ITU materials database), rather than being keyed by XML id.
        assert_eq!(
            material_names,
            vec!["itu_glass".to_string(), "itu_concrete".to_string()]
        );
    }

    #[test]
    fn load_xml_keeps_generic_name_for_thickness_less_material_despite_collision() {
        let dir = tempdir().expect("failed to create temporary directory");

        fs::write(dir.path().join("mesh.obj"), TRIANGLE_OBJ).expect("failed to write obj file");

        let xml = format!(
            r#"<scene version="2.1.0">
                <bsdf type="itu-radio-material" id="window1">
                    <string name="type" value="glass"/>
                    <float name="thickness" value="0.01"/>
                </bsdf>
                <bsdf type="itu-radio-material" id="window2">
                    <string name="type" value="glass"/>
                    <float name="thickness" value="0.05"/>
                </bsdf>
                <bsdf type="itu-radio-material" id="window3">
                    <string name="type" value="glass"/>
                </bsdf>
                {}{}{}
            </scene>"#,
            shape_xml("window1"),
            shape_xml("window2"),
            shape_xml("window3"),
        );

        let scene_file = dir.path().join("scene.xml");
        fs::write(&scene_file, xml).expect("failed to write scene file");

        let material_names: Vec<String> = Python::with_gil(|py| {
            let scene = Scene::load_xml(&scene_file).expect("scene should load");
            let py_scene =
                Bound::new(py, scene).expect("failed to wrap the scene in a Python object");
            let py_mesh = py_scene
                .getattr("mesh")
                .expect("scene should have a `mesh` attribute");
            py_mesh
                .getattr("material_names")
                .expect("mesh should have a `material_names` attribute")
                .extract()
                .expect("`material_names` should be extractable as Vec<String>")
        });

        // 'window1'/'window2' disagree on thickness and are kept
        // distinguishable under their own ids, but 'window3' has no
        // 'thickness' override of its own: it must keep the shared generic
        // name, since 'differt.em._material._populate_materials' never
        // creates an id-keyed entry for a thickness-less material (it would
        // otherwise have no matching entry on the Python side at all).
        assert_eq!(
            material_names,
            vec![
                "window1".to_string(),
                "window2".to_string(),
                "itu_glass".to_string(),
            ]
        );
    }
}

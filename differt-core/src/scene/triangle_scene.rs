use std::path::PathBuf;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use super::sionna::{Material, SionnaScene};
use crate::geometry::triangle_mesh::TriangleMesh;

/// A scene that contains triangle meshes and corresponding materials.
#[derive(Clone)]
#[pyclass(get_all)]
struct TriangleScene {
    /// list[differt_core.geometry.triangle_mesh.TriangleMesh]: The meshes.
    meshes: Vec<TriangleMesh>,
    /// list[differt_core.scene.sionna.Material]: The mesh materials.
    materials: Vec<Material>,
}

#[pymethods]
impl TriangleScene {
    /// Load a scene from a Sionna-compatible XML file.
    ///
    /// Args:
    ///     file (str): The path to the XML file.
    ///
    /// Returns
    ///     TriangleScene: The corresponding scene.
    #[classmethod]
    fn load_xml(cls: &Bound<'_, PyType>, file: &str) -> PyResult<Self> {
        // TODO: create a Rust variant without PyType?
        let sionna_scene_py_type = PyType::new_bound::<SionnaScene>(cls.py());
        let sionna = SionnaScene::load_xml(&sionna_scene_py_type, file)?;

        let path = PathBuf::from(file);
        let folder = path.parent().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Could not determine parent folder of file: {}",
                file
            ))
        })?;

        let mut meshes = Vec::with_capacity(sionna.shapes.len());
        let mut materials = Vec::with_capacity(sionna.shapes.len());

        let triangle_mesh_py_type = PyType::new_bound::<TriangleMesh>(cls.py());

        for (_, shape) in sionna.shapes.into_iter() {
            let mesh_file_path = folder.join(shape.file);
            let mesh_file = mesh_file_path.to_str().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Could not convert path {mesh_file_path:?} to valid unicode string"
                ))
            })?;
            let mesh = match shape.r#type.as_str() {
                "obj" => TriangleMesh::load_obj(&triangle_mesh_py_type, mesh_file)?,
                "ply" => TriangleMesh::load_ply(&triangle_mesh_py_type, mesh_file)?,
                ty => {
                    log::warn!("Unsupported shape type {ty}, skipping.");
                    continue;
                },
            };
            let material = sionna
                .materials
                .get(&shape.material_id)
                .cloned()
                .unwrap_or_default();
            meshes.push(mesh);
            materials.push(material);
        }
        Ok(Self { meshes, materials })
    }
}

#[pymodule]
pub(crate) fn triangle_scene(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TriangleScene>()?;
    Ok(())
}

use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;

use crate::geometry::triangle_mesh::TriangleMesh;

use super::sionna::SionnaScene;

#[pyclass]
struct TriangleScene {
    meshes: HashMap<String, TriangleMesh>,
}

#[pymethods]
impl TriangleScene {
    /// Load a triangle scene from a XML file.
    ///
    /// This method uses
    /// :meth:`SionnaScene.load_xml<differt.scene.sionna.SionnaScene.load_xml>`
    /// internally.
    #[classmethod]
    fn load_xml(cls: &PyType, file: &str) -> PyResult<Self> {
        // TODO: create a Rust variant without PyType?
        let sionna = SionnaScene::load_xml(cls, file)?;

        let path = PathBuf::from(file);
        let folder = path.parent().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Could not determine parent folder of file: {}",
                file
            ))
        })?;

        let mut meshes = HashMap::with_capacity(sionna.shapes.len());

        for (id, shape) in sionna.shapes.into_iter() {
            let mesh_file_path = folder.join(shape.file);
            let mesh_file = mesh_file_path.to_str().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Could not convert path {mesh_file_path:?} to valid unicode string"
                ))
            })?;
            let mesh = match shape.r#type.as_str() {
                "obj" => TriangleMesh::load_obj(cls, mesh_file)?,
                "ply" => TriangleMesh::load_ply(cls, mesh_file)?,
                ty => {
                    log::warn!("Unsupported shape type {ty}, skipping.");
                    continue;
                },
            };
            meshes.insert(id, mesh);
        }
        Ok(Self { meshes })
    }
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "triangle_scene")?;
    m.add_class::<TriangleScene>()?;

    Ok(m)
}

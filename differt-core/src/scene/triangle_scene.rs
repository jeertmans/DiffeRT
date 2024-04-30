use std::{collections::HashMap, ops::Range, path::PathBuf};

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PySlice, PyType},
};

use super::sionna::SionnaScene;
use crate::geometry::triangle_mesh::TriangleMesh;

/// TODO.
#[derive(Clone)]
#[pyclass]
struct TriangleScene {
    #[pyo3(get)]
    mesh: TriangleMesh,
    mesh_ids: HashMap<String, Range<usize>>,
}

#[pymethods]
impl TriangleScene {
    #[getter]
    fn mesh_ids<'py>(&self, py: Python<'py>) -> HashMap<String, Bound<'py, PySlice>> {
        self.mesh_ids
            .iter()
            .map(|(id, range)| {
                (
                    id.clone(),
                    PySlice::new_bound(py, range.start as _, range.end as _, 1),
                )
            })
            .collect()
    }
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

        let mut mesh = TriangleMesh::default();
        let mut mesh_ids = HashMap::with_capacity(sionna.shapes.len());

        let triangle_mesh_py_type = PyType::new_bound::<TriangleMesh>(cls.py());

        let mut start = 0;

        for (id, shape) in sionna.shapes.into_iter() {
            let mesh_file_path = folder.join(shape.file);
            let mesh_file = mesh_file_path.to_str().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Could not convert path {mesh_file_path:?} to valid unicode string"
                ))
            })?;
            let mut other_mesh = match shape.r#type.as_str() {
                "obj" => TriangleMesh::load_obj(&triangle_mesh_py_type, mesh_file)?,
                "ply" => TriangleMesh::load_ply(&triangle_mesh_py_type, mesh_file)?,
                ty => {
                    log::warn!("Unsupported shape type {ty}, skipping.");
                    continue;
                },
            };
            mesh.append(&mut other_mesh);
            let end = mesh.triangles.len();
            mesh_ids.insert(id, start..end);
            start = end;
        }
        Ok(Self { mesh, mesh_ids })
    }
}

#[pymodule]
pub(crate) fn triangle_scene(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TriangleScene>()?;
    Ok(())
}

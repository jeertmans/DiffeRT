use std::path::PathBuf;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use super::sionna::SionnaScene;
use crate::geometry::mesh::Mesh;

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
    ///     file (str): The path to the XML file.
    ///
    /// Returns:
    ///     Scene: The corresponding scene.
    #[classmethod]
    fn load_xml(cls: &Bound<'_, PyType>, file: &str) -> PyResult<Self> {
        // TODO: create a Rust variant without PyType?
        let sionna_scene_py_type = PyType::new::<SionnaScene>(cls.py());
        let sionna = SionnaScene::load_xml(&sionna_scene_py_type, file)?;

        let path = PathBuf::from(file);
        let folder = path.parent().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Could not determine parent folder of file: {}",
                file
            ))
        })?;

        let mut mesh = Mesh::default();

        let mesh_py_type = PyType::new::<Mesh>(cls.py());

        for (_, shape) in sionna.shapes.into_iter() {
            let mesh_file_path = folder.join(shape.file);
            let mesh_file = mesh_file_path.to_str().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Could not convert path {mesh_file_path:?} to valid unicode string"
                ))
            })?;
            let mut other_mesh = match shape.r#type.as_str() {
                "obj" => Mesh::load_obj(&mesh_py_type, mesh_file)?,
                "ply" => Mesh::load_ply(&mesh_py_type, mesh_file)?,
                ty => {
                    log::warn!("Unsupported shape type {ty}, skipping.");
                    continue;
                },
            };

            let material = sionna.materials.get(&shape.material_id);

            let color = material.map(|mat| mat.color);

            let material_name = material.map(|mat| mat.name.clone());

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

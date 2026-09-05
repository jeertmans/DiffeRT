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
            let material_name = material.map(|mat| {
                if non_uniform_names.contains(&mat.name) {
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

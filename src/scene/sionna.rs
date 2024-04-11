use std::{collections::HashMap, fs::File, io::BufReader};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};
use serde::{de, Deserialize};

/// A scene as loaded from a Sionna-compatible
/// XML file.
///
/// Only a subset of the XML file is actually used.
///
/// This class is useless unless converted
/// in another scene type, like
/// :class:`TriangleScene<differt.scene.triangle_scene.TriangleScene>`.
#[pyclass(get_all)]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SionnaScene {
    /// A mapping between material IDs and actual
    /// materials.
    ///
    /// Currently, only BSDF materials are used.
    #[serde(rename = "bsdf", deserialize_with = "deserialize_materials")]
    pub(crate) materials: HashMap<String, Material>,
    /// A mapping between shape IDs and actual
    /// materials.
    ///
    /// Currently, only shapes from files are supported.
    ///
    /// Also, face normals attribute is ignored, as normals are always
    /// recomputed.
    #[serde(rename = "shape", deserialize_with = "deserialize_shapes")]
    pub(crate) shapes: HashMap<String, Shape>,
}

fn deserialize_materials<'de, D>(deserializer: D) -> Result<HashMap<String, Material>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Vec::<Material>::deserialize(deserializer).map(|v| {
        let mut map = HashMap::with_capacity(v.len());

        for material in v {
            if let Some(material) = map.insert(material.id.clone(), material) {
                log::warn!(
                    "duplicate material ID, the latter was removed '{:?}'",
                    material
                );
            }
        }

        map
    })
}

fn deserialize_shapes<'de, D>(deserializer: D) -> Result<HashMap<String, Shape>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Vec::<Shape>::deserialize(deserializer).map(|v| {
        let mut map = HashMap::with_capacity(v.len());

        for shape in v {
            if let Some(shape) = map.insert(shape.id.clone(), shape) {
                log::warn!("duplicate shape ID, the latter was removed '{:?}'", shape);
            }
        }

        map
    })
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Material {
    #[serde(rename(deserialize = "@type"))]
    pub(crate) r#type: String,
    #[serde(rename(deserialize = "@id"))]
    pub(crate) id: String,
}

/// A shape, that is part of a scene.
#[pyclass(get_all)]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Shape {
    /// The type of the shape file.
    ///
    /// E.g., `ply` for Stanford PLY format.
    #[serde(rename(deserialize = "@type"))]
    pub(crate) r#type: String,
    /// The shape ID.
    ///
    /// It should be unique (in a given scene).
    #[serde(rename(deserialize = "@id"))]
    pub(crate) id: String,
    /// The path to the shape file.
    ///
    /// This path is relative to the scene config file.
    #[serde(rename(deserialize = "string"), deserialize_with = "deserialize_file")]
    pub(crate) file: String,
    /// The material ID attached to this object.
    #[serde(
        rename(deserialize = "ref"),
        deserialize_with = "deserialize_material_id"
    )]
    pub(crate) material_id: String,
}

fn deserialize_file<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: de::Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct FileString {
        #[serde(rename(deserialize = "@name"))]
        name: String,
        #[serde(rename(deserialize = "@value"))]
        value: String,
    }

    let FileString { name, value } = FileString::deserialize(deserializer)?;

    if name == "filename" {
        Ok(value)
    } else {
        Err(de::Error::custom(
            "value of <string> element must be 'filename'",
        ))
    }
}

fn deserialize_material_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: de::Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct MaterialId {
        #[serde(rename(deserialize = "@id"))]
        id: String,
    }

    MaterialId::deserialize(deserializer).map(|mat_id| mat_id.id)
}

#[pymethods]
impl SionnaScene {
    /// Load a Sionna scene from a XML file.
    #[classmethod]
    pub(crate) fn load_xml(_: &PyType, file: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(file)?);
        quick_xml::de::from_reader(input).map_err(|err| {
            PyValueError::new_err(format!("An error occurred while reading XML file: {}", err))
        })
    }
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "sionna")?;
    m.add_class::<Material>()?;
    m.add_class::<SionnaScene>()?;
    m.add_class::<Shape>()?;

    Ok(m)
}

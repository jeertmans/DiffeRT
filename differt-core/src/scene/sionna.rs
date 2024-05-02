// TODO: fix attribute type hint in docs.
use std::{collections::HashMap, fs::File, io::BufReader};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};
use serde::{de, Deserialize};

/// A scene as loaded from a Sionna-compatible
/// XML file.
///
/// Only a subset of the XML file is actually used.
///
/// This class is useless unless converted
/// to another scene type, like
/// :class:`TriangleScene<differt.scene.triangle_scene.TriangleScene>`.
///
/// Warning:
///     Currently, the ``'etoile'`` scene from Sionna cannot be loaded
///     properly, as the material properties are encoded differently.
///
///     We are still thinking of a better way to parse those XML files,
///     please reach out us if you would like to help!
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

/// A basic material, that can be linked to EM properties.
#[pyclass(get_all)]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Material {
    /// The material ID.
    ///
    /// This can be, e.g., an ITU identifier.
    ///
    /// # type: str
    #[serde(rename(deserialize = "@id"))]
    pub(crate) id: String,
    /// The material color, used when plotted.
    #[serde(rename(deserialize = "bsdf"), deserialize_with = "deserialize_rgb")]
    pub(crate) rgb: (f32, f32, f32),
}

fn deserialize_rgb<'de, D>(deserializer: D) -> Result<(f32, f32, f32), D::Error>
where
    D: de::Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct Bsdf {
        rgb: Rgb,
    }

    #[derive(Deserialize)]
    struct Rgb {
        #[serde(rename(deserialize = "@value"))]
        value: String,
    }

    let Bsdf { rgb } = Bsdf::deserialize(deserializer)?;

    match rgb
        .value
        .split_ascii_whitespace()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [r_str, g_str, b_str] => {
            let r = r_str.parse().map_err(de::Error::custom)?;
            let g = g_str.parse().map_err(de::Error::custom)?;
            let b = b_str.parse().map_err(de::Error::custom)?;
            Ok((r, g, b))
        },
        _ => {
            Err(de::Error::custom(
                "value of <rgb> element must contain three floats",
            ))
        },
    }
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
    ///
    /// Args:
    ///     file: The path to the XML file.
    ///
    /// Return:
    ///     The corresponding scene.
    #[classmethod]
    pub(crate) fn load_xml(_: &Bound<'_, PyType>, file: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(file)?);
        quick_xml::de::from_reader(input).map_err(|err| {
            PyValueError::new_err(format!("An error occurred while reading XML file: {}", err))
        })
    }
}

#[pymodule]
pub(crate) fn sionna(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Material>()?;
    m.add_class::<SionnaScene>()?;
    m.add_class::<Shape>()?;
    Ok(())
}

use std::{fs::File, io::BufReader};

use indexmap::IndexMap;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};
use serde::{Deserialize, de};

/// A scene as loaded from a Sionna-compatible XML file.
///
/// Only a subset of the XML file is actually used.
///
/// This class is useless unless converted
/// to another scene type, like
/// :class:`TriangleScene<differt.scene.TriangleScene>`.
///
/// Warning:
///     We are still open to better ways to parse those XML files,
///     please reach out us if you would like to help!
#[pyclass(get_all)]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SionnaScene {
    /// dict[str, Material]: A mapping between material IDs and actual materials.
    ///
    /// Currently, only BSDF materials are used.
    #[serde(rename = "bsdf", deserialize_with = "deserialize_materials")]
    pub(crate) materials: IndexMap<String, Material>,
    /// dict[str, Shape]: A mapping between shape IDs and actual shapes.
    ///
    /// Currently, only shapes from files are supported.
    ///
    /// Also, any face normals attribute is ignored, as normals are
    /// recomputed using JAX arrays in the :mod:`differt` module.
    #[serde(rename = "shape", deserialize_with = "deserialize_shapes")]
    pub(crate) shapes: IndexMap<String, Shape>,
}

fn deserialize_materials<'de, D>(deserializer: D) -> Result<IndexMap<String, Material>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Vec::<Material>::deserialize(deserializer).map(|v| {
        let mut map = IndexMap::with_capacity(v.len());

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

fn deserialize_shapes<'de, D>(deserializer: D) -> Result<IndexMap<String, Shape>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Vec::<Shape>::deserialize(deserializer).map(|v| {
        let mut map = IndexMap::with_capacity(v.len());

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
#[derive(Clone, Debug, Default)]
pub(crate) struct Material {
    /// str: The material name.
    ///
    /// This can be, e.g., an ITU identifier.
    pub(crate) name: String,
    /// str: The material ID.
    ///
    /// This is used to match the material id used in shapes.
    pub(crate) id: String,
    /// tuple[float, float, float]: The material color, used for plotting.
    ///
    /// The color is obtained from a---possibly-nested---``<rgb>`` element,
    /// or from a ``<string>`` element with a ``type`` attribute. In the latter
    /// case, the color is chosen from a predefined list of colors.
    pub(crate) color: [f32; 3],
    /// typing.Optional[float]: The thickness of the material.
    pub(crate) thickness: Option<f32>,
}

fn deserialize_rgb<'de, D>(deserializer: D) -> Result<[f32; 3], D::Error>
where
    D: de::Deserializer<'de>,
{
    let value = <String>::deserialize(deserializer)?;

    match value
        .split_ascii_whitespace()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [r_str, g_str, b_str] => {
            let r = r_str.parse().map_err(de::Error::custom)?;
            let g = g_str.parse().map_err(de::Error::custom)?;
            let b = b_str.parse().map_err(de::Error::custom)?;
            Ok([r, g, b])
        },
        _ => {
            Err(de::Error::custom(
                "value of <rgb> element must contain three floats",
            ))
        },
    }
}

impl<'de> Deserialize<'de> for Material {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        struct Rgb {
            #[serde(rename(deserialize = "@value"), deserialize_with = "deserialize_rgb")]
            color: [f32; 3],
        }

        #[derive(Debug, Deserialize)]
        #[serde(rename_all = "snake_case")]
        struct Bsdf {
            #[serde(rename = "rgb")]
            rgb: Rgb,
        }

        #[derive(Debug)]
        enum Type {
            Struct { value: String },
        }

        quick_xml::impl_deserialize_for_internally_tagged_enum! {
            Type, "@name",
            ("type"    => Struct {
                #[serde(rename = "@value")]
                 value: String }),
        }

        #[derive(Debug)]
        enum Thickness {
            Struct { value: f32 },
        }

        quick_xml::impl_deserialize_for_internally_tagged_enum! {
            Thickness, "@name",
            ("thickness"    => Struct {
                #[serde(rename = "@value")]
                value: f32,
            }),
        }

        #[derive(Debug)]
        enum RawMaterial {
            TwoSided {
                id: String,
                bsdf: Bsdf,
                //thickness: Option<f32>,
            },
            ItuRadioMaterial {
                id: String,
                r#type: Type,
                thickness: Option<Thickness>,
            },
        }

        quick_xml::impl_deserialize_for_internally_tagged_enum! {
            RawMaterial, "@type",
            ("twosided"    => TwoSided {
                #[serde(rename = "@id")]
                id: String,
                bsdf: Bsdf,
            }),
            ("itu-radio-material"  => ItuRadioMaterial {
                #[serde(rename = "@id")]
                id: String,
                #[serde(rename = "string")]
                r#type: Type,
                #[serde(skip)]
                thickness: Option<Thickness>,
            }),
        }

        let raw_mat = RawMaterial::deserialize(deserializer)?;

        match raw_mat {
            RawMaterial::TwoSided {
                id,
                bsdf: Bsdf { rgb: Rgb { color } },
            } => {
                Ok(Material {
                    name: id.strip_prefix("mat-").unwrap_or(&id).to_string(),
                    id,
                    color,
                    thickness: None,
                })
            },
            RawMaterial::ItuRadioMaterial {
                id,
                r#type: Type::Struct { value: r#type },
                thickness,
            } => {
                let color = match r#type.as_str() {
                    // Copied from Sionna-RT's code, to match their colors:
                    // https://github.com/NVlabs/sionna-rt/blob/main/src/sionna/rt/radio_materials/itu_material.py
                    // v1.0.0
                    "marble" => [0.701, 0.644, 0.485],
                    "concrete" => [0.539, 0.539, 0.539],
                    "wood" => [0.266, 0.109, 0.060],
                    "metal" => [0.220, 0.220, 0.254],
                    "brick" => [0.402, 0.112, 0.087],
                    "glass" => [0.168, 0.139, 0.509],
                    "floorboard" => [0.539, 0.386, 0.025],
                    "ceiling_board" => [0.376, 0.539, 0.117],
                    "chipboard" => [0.509, 0.159, 0.323],
                    "plasterboard" => [0.051, 0.539, 0.133],
                    "plywood" => [0.136, 0.076, 0.539],
                    "very_dry_ground" => [0.539, 0.319, 0.223],
                    "medium_dry_ground" => [0.539, 0.181, 0.076],
                    "wet_ground" => [0.539, 0.027, 0.147],
                    _ => {
                        log::warn!(
                            "unknown material type: {type:#?}, using default color, i.e., black",
                        );
                        [0.0, 0.0, 0.0]
                    },
                };
                Ok(Material {
                    name: format!("itu_{type}"),
                    id,
                    color,
                    thickness: thickness.map(|t| {
                        match t {
                            Thickness::Struct { value: thickness } => thickness,
                        }
                    }),
                })
            },
        }
    }
}

/// A shape, that is part of a scene.
#[pyclass(get_all)]
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Shape {
    /// str: The type of the shape file.
    ///
    /// E.g., `ply` for Stanford PLY format.
    #[serde(rename(deserialize = "@type"))]
    pub(crate) r#type: String,
    /// str: The shape ID.
    ///
    /// It should be unique (in a given scene).
    #[serde(rename(deserialize = "@id"))]
    pub(crate) id: String,
    /// str: The path to the shape file.
    ///
    /// This path is relative to the scene config file.
    #[serde(rename(deserialize = "string"), deserialize_with = "deserialize_file")]
    pub(crate) file: String,
    /// str: The material ID attached to this object.
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
    ///     file (str): The path to the XML file.
    ///
    /// Returns:
    ///     SionnaScene: The corresponding scene.
    #[classmethod]
    pub(crate) fn load_xml(_: &Bound<'_, PyType>, file: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(file)?);
        quick_xml::de::from_reader(input).map_err(|err| {
            PyValueError::new_err(format!(
                "An error occurred while reading XML file {file:#?}: {}",
                err
            ))
        })
    }
}

#[cfg(not(tarpaulin_include))]
#[pymodule]
pub(crate) fn sionna(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Material>()?;
    m.add_class::<SionnaScene>()?;
    m.add_class::<Shape>()?;
    Ok(())
}

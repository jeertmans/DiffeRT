use std::{fs::File, io::BufReader};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[pyclass]
struct SionnaScene {
    #[serde(rename = "bsdf")]
    #[pyo3(get)]
    materials: Vec<Material>,
    #[serde(rename = "shape")]
    #[pyo3(get)]
    objects: Vec<Object>,
}

#[derive(Clone, Debug, Deserialize)]
#[pyclass]
struct Material {
    #[serde(rename(deserialize = "@type"))]
    _type: String,
    #[serde(rename(deserialize = "@id"))]
    id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[pyclass]
struct Object {
    #[serde(rename(deserialize = "@type"))]
    _type: String,
    #[serde(rename(deserialize = "@id"))]
    id: String,
    #[serde(rename(deserialize = "string"))]
    file: String,
    #[serde(rename(deserialize = "ref"))]
    material_id: String,
}

#[pymethods]
impl SionnaScene {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    #[classmethod]
    fn load_xml(_: &PyType, filename: &str) -> PyResult<Self> {
        let input = BufReader::new(File::open(filename)?);
        quick_xml::de::from_reader(input).map_err(|err| {
            PyValueError::new_err(format!("An error occurred while reading XML file: {}", err))
        })
    }
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "sionna")?;
    m.add_class::<SionnaScene>()?;

    Ok(m)
}

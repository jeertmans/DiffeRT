use pyo3::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
#[pyclass]
struct TriangleScene {
    objects: Vec<usize>,
}

#[pymethods]
impl TriangleScene {}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "triangle_scene")?;
    m.add_class::<TriangleScene>()?;

    Ok(m)
}

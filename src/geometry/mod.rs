use pyo3::prelude::*;

pub mod triangle_mesh;

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "geometry")?;
    m.add_submodule(triangle_mesh::create_module(py)?)?;
    Ok(m)
}

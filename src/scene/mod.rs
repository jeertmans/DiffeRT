use pyo3::prelude::*;

pub mod sionna;
pub mod triangle_scene;

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "scene")?;
    m.add_submodule(sionna::create_module(py)?)?;
    m.add_submodule(triangle_scene::create_module(py)?)?;
    Ok(m)
}

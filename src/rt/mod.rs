use pyo3::prelude::*;

pub mod graph;
pub mod utils;

pub(crate) fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = pyo3::prelude::PyModule::new_bound(py, "rt")?;

    m.add_submodule(&utils::create_module(py)?)?;
    m.add_submodule(&graph::create_module(py)?)?;

    Ok(m)
}

use pyo3::prelude::*;

pub mod rt;

const VERSION: &str = "0.0.5";

/// Core of DiffeRT module, implemented in Rust.
#[pymodule]
fn differt_core(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_submodule(rt::create_module(py)?)?;
    Ok(())
}

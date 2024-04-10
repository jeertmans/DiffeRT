use pyo3::prelude::*;

pub mod geometry;
pub mod rt;
pub mod scene;

/// Core of DiffeRT module, implemented in Rust.
#[pymodule]
fn _core(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    let mut version = env!("CARGO_PKG_VERSION").to_string();
    version = version.replace("-alpha", "a").replace("-beta", "b");
    m.add("__version__", version)?;
    m.add_submodule(geometry::create_module(py)?)?;
    m.add_submodule(rt::create_module(py)?)?;
    m.add_submodule(scene::create_module(py)?)?;
    Ok(())
}

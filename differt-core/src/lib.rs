use pyo3::{prelude::*, wrap_pymodule};

pub mod geometry;
pub mod rt;
pub mod scene;

/// Core of DiffeRT module, implemented in Rust.
#[pymodule]
fn _lowlevel(m: Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    let mut version = env!("CARGO_PKG_VERSION").to_string();
    version = version.replace("-alpha", "a").replace("-beta", "b");
    m.add("__version__", version)?;
    m.add_wrapped(wrap_pymodule!(geometry::geometry))?;
    m.add_wrapped(wrap_pymodule!(rt::rt))?;
    m.add_wrapped(wrap_pymodule!(scene::scene))?;
    Ok(())
}

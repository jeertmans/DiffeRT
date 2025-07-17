use pyo3::{prelude::*, wrap_pymodule};

pub mod geometry;
pub mod rt;
pub mod scene;

/// Core of DiffeRT module, implemented in Rust.
#[cfg(not(tarpaulin_include))]
#[pymodule(gil_used = false)]
fn _differt_core(m: Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    let mut version = env!("CARGO_PKG_VERSION").to_string();
    version = version.replace("-alpha", "a").replace("-beta", "b");
    m.add("__version__", version)?;
    let version_info = (
        env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap(),
        env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap(),
        env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap(),
    );
    m.add("__version_info__", version_info)?;
    m.add_wrapped(wrap_pymodule!(geometry::geometry))?;
    m.add_wrapped(wrap_pymodule!(rt::rt))?;
    m.add_wrapped(wrap_pymodule!(scene::scene))?;
    Ok(())
}

use pyo3::{prelude::*, wrap_pymodule};

pub mod sionna;
pub mod triangle_scene;

#[cfg(not(tarpaulin_include))]
#[pymodule(gil_used = false)]
pub(crate) fn scene(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(sionna::sionna))?;
    m.add_wrapped(wrap_pymodule!(triangle_scene::triangle_scene))?;
    Ok(())
}

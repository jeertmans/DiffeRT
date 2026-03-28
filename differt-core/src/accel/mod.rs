use pyo3::{prelude::*, wrap_pymodule};

pub mod bvh;

#[cfg(not(tarpaulin_include))]
#[pymodule(gil_used = false)]
pub(crate) fn accel(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(bvh::bvh))?;
    Ok(())
}

use pyo3::{prelude::*, wrap_pymodule};

pub mod graph;

#[cfg(not(tarpaulin_include))]
#[pymodule(gil_used = false)]
pub(crate) fn rt(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(graph::graph))?;
    Ok(())
}

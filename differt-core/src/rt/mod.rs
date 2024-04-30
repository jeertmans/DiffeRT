use pyo3::{prelude::*, wrap_pymodule};

pub mod graph;
pub mod utils;

#[pymodule]
pub(crate) fn rt(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(graph::graph))?;
    m.add_wrapped(wrap_pymodule!(utils::utils))?;
    Ok(())
}

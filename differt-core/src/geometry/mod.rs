use pyo3::{prelude::*, wrap_pymodule};

pub mod graph;
pub mod mesh;
pub mod scene;
pub mod sionna;

#[cfg(not(tarpaulin_include))]
#[pymodule(gil_used = false)]
pub(crate) fn geometry(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(graph::graph))?;
    m.add_wrapped(wrap_pymodule!(mesh::mesh))?;
    m.add_wrapped(wrap_pymodule!(scene::scene))?;
    m.add_wrapped(wrap_pymodule!(sionna::sionna))?;
    Ok(())
}

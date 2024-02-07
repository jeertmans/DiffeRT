use numpy::{
    ndarray::{parallel::prelude::*, s, Array2, ArrayView2, Axis},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2,
};
use pyo3::prelude::*;

use super::graph::{collect_paths_in_array, CompleteGraph, DiGraph, IntoAllPathsIterator};

/// Generate an array of all path candidates (assuming fully connected
/// primitives).
#[pyfunction]
pub fn generate_all_path_candidates(
    py: Python<'_>,
    num_primitives: usize,
    order: usize,
) -> &PyArray2<usize> {
    // TODO: should we really transpose?
    let mut graph: DiGraph = CompleteGraph::new(num_primitives).into();
    let (from, to) = graph.insert_from_and_to_nodes(true);
    let paths = graph
        .all_paths(from, to, order + 2)
        .map(|path| path[1..path.len() - 1].to_vec());
    let path_candidates = collect_paths_in_array(paths, order);
    path_candidates.t().to_owned().into_pyarray(py)
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "utils")?;
    m.add_function(wrap_pyfunction!(generate_all_path_candidates, m)?)?;

    Ok(m)
}

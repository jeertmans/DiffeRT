use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use super::graph::{
    complete::{AllPathsFromCompleteGraphChunksIter, AllPathsFromCompleteGraphIter, CompleteGraph},
    PathsIterator,
};

/// Generate an array of all path candidates (assuming fully connected
/// primitives).
#[pyfunction]
pub fn generate_all_path_candidates(
    py: Python<'_>,
    num_primitives: usize,
    order: usize,
) -> &PyArray2<usize> {
    let graph = CompleteGraph::new(num_primitives);
    let from = num_primitives;
    let to = num_primitives + 1;
    graph
        .all_paths(from, to, order + 2, false)
        .collect_array()
        .into_pyarray(py)
}

/// Iterator variant of eponym function.
#[pyfunction]
pub fn generate_all_path_candidates_iter(
    num_primitives: usize,
    order: usize,
) -> AllPathsFromCompleteGraphIter {
    let graph = CompleteGraph::new(num_primitives);
    let from = num_primitives;
    let to = num_primitives + 1;
    graph.all_paths(from, to, order + 2, false)
}

/// Iterator variant of eponym function,
/// grouped in chunks of size of max. ``chunk_size``.
#[pyfunction]
pub fn generate_all_path_candidates_chunks_iter(
    num_primitives: usize,
    order: usize,
    chunk_size: usize,
) -> AllPathsFromCompleteGraphChunksIter {
    let graph = CompleteGraph::new(num_primitives);
    let from = num_primitives;
    let to = num_primitives + 1;
    graph.all_paths_array_chunks(from, to, order + 2, false, chunk_size)
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "utils")?;
    m.add_function(wrap_pyfunction!(generate_all_path_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(generate_all_path_candidates_iter, m)?)?;
    m.add_function(wrap_pyfunction!(
        generate_all_path_candidates_chunks_iter,
        m
    )?)?;

    Ok(m)
}

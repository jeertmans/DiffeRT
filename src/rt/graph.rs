use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

/// A simple directed graph.
struct Graph {}

impl<'source> FromPyObject<'source> for Graph {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let array = PyReadonlyArray2::<'source, bool>::extract(ob)?;
        Ok(Graph {})
    }
}

#[pyclass]
struct AllPathsFromGraphIterator {
    graph: Graph,
    from: usize,
    to: usize,
    depth: usize,
}

impl AllPathsFromGraphIterator {
    pub fn new(graph: Graph, from: usize, to: usize, depth: usize) -> Self {
        Self {
            graph,
            from,
            to,
            depth,
        }
    }
}

impl Iterator for AllPathsFromGraphIterator {
    type Item = Vec<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/*
fn all_paths_array_from_graph(graph: usize, depth: u32) -> usize {
    todo!()
}*/

#[pyfunction]
fn all_paths_iter_from_graph(
    graph: Graph,
    from: usize,
    to: usize,
    depth: usize,
) -> AllPathsFromGraphIterator {
    AllPathsFromGraphIterator::new(graph, from, to, depth)
}
/*
/// Generate all paths of length `depth`.
///
/// For every node,
fn all_paths_array_from_num_nodes(num_nodes: usize, depth: u32) -> usize {
    todo!()
}
fn all_paths_array_from_num_nodes(num_nodes: usize, depth: u32) -> usize {
    todo!()
}
*/

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "graph")?;
    m.add_function(wrap_pyfunction!(all_paths_iter_from_graph, m)?)?;

    Ok(m)
}

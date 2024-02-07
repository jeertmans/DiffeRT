use core::ops::Range;
use std::{collections::VecDeque, iter::FlatMap};

use numpy::{
    ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView2, Axis},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2,
};
use pyo3::prelude::*;

/// An iterator over paths that have the same depth.
pub trait PathsIterator: Iterator<Item = Vec<usize>> {
    /// Hint about path depth.
    ///
    /// If not specified, the first path is used to
    /// determine the depth. This is useful if
    /// one needs to have the second axis of
    /// the array be equal to `depth()`, even
    /// in the event of an empty iterator.
    #[inline]
    fn depth(&self) -> Option<usize> {
        None
    }

    /// Collect all paths into a 2D array.
    ///
    /// # Panicking
    ///
    /// If not all paths have the same depth.
    ///
    /// If it cannot allocate enough memory.
    fn as_array(&mut self) -> Array2<usize> {
        let mut flat_vec = Vec::new();
        let mut num_paths = 0;
        let depth = match self.depth() {
            Some(depth) => depth,
            None => {
                match self.next() {
                    Some(path) => {
                        num_paths += 1;
                        flat_vec.extend_from_slice(path.as_ref());
                        path.len() // Depth from first path
                    },
                    None => return Array2::default((0, 0)), // No path, returning
                }
            },
        };

        self.for_each(|path| {
            num_paths += 1;
            flat_vec.extend_from_slice(path.as_ref());
        });
        Array2::from_shape_vec((num_paths, depth), flat_vec).unwrap()
    }
}

pub struct PathsIter<T> {
    iter: T,
    maybe_depth: Option<usize>,
}

impl<T: Iterator<Item = Vec<usize>>> PathsIter<T> {
    fn from_iter_and_depth(iter: T, depth: usize) -> Self {
        Self {
            iter,
            maybe_depth: Some(depth),
        }
    }
}

impl<T: Iterator<Item = Vec<usize>>> Iterator for PathsIter<T> {
    type Item = Vec<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<T: Iterator<Item = Vec<usize>>> PathsIterator for PathsIter<T> {
    fn depth(&self) -> Option<usize> {
        self.maybe_depth
    }
}

pub fn collect_paths_in_array<P, I>(paths: I, depth: usize) -> Array2<usize>
where
    P: AsRef<[usize]>,
    I: Iterator<Item = P>,
{
    let mut flat_vec = Vec::new();
    let mut num_paths = 0;
    paths.for_each(|path| {
        num_paths += 1;
        flat_vec.extend_from_slice(path.as_ref());
    });
    Array2::from_shape_vec((num_paths, depth), flat_vec).unwrap()
}

/*
trait CollectArray<T> {
    fn collect_array(&mut self)
}

impl CollectArray<T>
*/

pub trait IntoAllPathsIterator {
    /// Return the number of unique nodes.
    fn num_nodes(&self) -> usize;

    /// Return an iterator over all paths of length `depth`
    /// from node `from` to node `to`.
    fn all_paths(&self, from: usize, to: usize, depth: usize) -> impl Iterator<Item = Vec<usize>>;

    /// Collect all paths into a 2D array.
    ///
    /// # Panicking
    ///
    /// If not all paths have the same depth.
    ///
    /// If it cannot allocate enough memory.
    fn all_paths_array(&self, from: usize, to: usize, depth: usize) -> Array2<usize> {
        collect_paths_in_array(self.all_paths(from, to, depth), self.num_nodes())
    }

    /// Return an iterator over all paths between all pairs
    /// of `from` and `to` nodes.
    fn all_pairs_all_paths(&self, depth: usize) -> impl PathsIterator {
        let num_nodes = self.num_nodes();

        PathsIter::from_iter_and_depth(
            (0..num_nodes).flat_map(move |from| {
                (0..num_nodes).flat_map(move |to| self.all_paths(from, to, depth))
            }),
            depth,
        )
    }
}

/// A complete graph, i.e.,
/// a simple undirected graph in which every pair of
/// distinc nodes is connected by a unique edge.
#[pyclass]
#[derive(Clone, Debug)]
pub struct CompleteGraph {
    /// Number of nodes.
    num_nodes: usize,
}

#[pymethods]
impl CompleteGraph {
    #[new]
    pub fn new(num_nodes: usize) -> CompleteGraph {
        Self { num_nodes }
    }
}

impl IntoAllPathsIterator for CompleteGraph {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    fn all_paths(&self, from: usize, to: usize, depth: usize) -> impl Iterator<Item = Vec<usize>> {
        AllPathsFromCompleteGraphIter::new(self.num_nodes, from, to, depth)
    }
}

/// An iterator over all paths in a complete graph.
#[pyclass]
#[derive(Clone, Debug)]
struct AllPathsFromCompleteGraphIter {
    num_nodes: usize,
    from: usize,
    to: usize,
    depth: usize,
    path: Vec<usize>,
    done: bool,
}

/// Create a mapping from
/// [0, 1, ..., num_nodes - 1]
/// to
/// [0, 1, ..., from*, from + 1, ..., to*, to + 1, ..., num_nodes - 1]
///
/// This is useful to make path methods agnostic of the actual start and
/// end nodes.
///
/// *: if from > to, then swap the two variable names.
fn make_nodes_map(num_nodes: usize, from: usize, to: usize) -> Vec<usize> {
    let (min, max) = if from < to { (from, to) } else { (to, from) };
    let mut nodes_map: Vec<usize> = (0..num_nodes).collect();

    nodes_map[min..max].iter_mut().for_each(|i| *i = *i + 1);
    nodes_map[max..].iter_mut().for_each(|i| *i = *i + 2);

    nodes_map[num_nodes - 1] = to;

    if from != to {
        nodes_map[num_nodes - 2] = from;
    }

    nodes_map
}

impl AllPathsFromCompleteGraphIter {
    #[inline]
    pub fn new(num_nodes: usize, from: usize, to: usize, depth: usize) -> Self {
        if depth < 2 || (depth == 2 && from == to) || num_nodes < 2 {
            return Self {
                num_nodes,
                from,
                to,
                depth,
                path: vec![],
                done: true,
            };
        }

        let path = (0..depth).collect();
        let _nodes_maps = make_nodes_map(num_nodes, from, to);

        Self {
            num_nodes,
            from,
            to,
            depth,
            path,
            done: false,
        }
    }
}

impl Iterator for AllPathsFromCompleteGraphIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        todo!()
    }
}

impl PathsIterator for AllPathsFromCompleteGraphIter {
    fn depth(&self) -> Option<usize> {
        Some(self.depth)
    }
}

/// A directed graph.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DiGraph {
    /// List of list of edges,
    /// where edges[i] is the list of adjacent nodes
    /// of node i.
    edges_list: Vec<Vec<usize>>,
}

impl DiGraph {
    #[inline]
    pub fn get_adjacent_nodes(&self, node: usize) -> &[usize] {
        self.edges_list[node].as_ref()
    }

    pub fn from_adjacency_matrix(adjacency_matrix: &ArrayView2<bool>) -> Self {
        debug_assert!(
            adjacency_matrix.is_square(),
            "adjacency matrix must be square"
        );
        let edges_list = adjacency_matrix
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                row.indexed_iter()
                    .filter_map(|(index, &item)| if item { Some(index) } else { None })
                    .collect()
            })
            .collect();

        Self { edges_list }
    }

    /// Insert two additional nodes in the graph:
    ///
    /// a `from` node, that is connected to every other node in the graph;
    /// and a `to` node, where every other node is connected to this node.
    ///
    /// If `direct_path` is `true`, then the `from` node is connected to the
    /// `to` node.
    ///
    /// Return the indices of the two nodes in the graph.
    pub fn insert_from_and_to_nodes(&mut self, direct_path: bool) -> (usize, usize) {
        let from = self.edges_list.len();
        let to = from + 1;

        // Every node is connected to `to`.
        self.edges_list.iter_mut().for_each(|edges| edges.push(to));

        // `from` is connected to every node except itself
        let mut from_edges: Vec<usize> = (0..from).collect();

        if direct_path {
            from_edges.push(to);
        }

        self.edges_list.push(from_edges);

        // `to` is not connected to any node
        self.edges_list.push(vec![]);

        self.get_adjacent_nodes(from);
        self.get_adjacent_nodes(to);

        (from, to)
    }
}

impl From<CompleteGraph> for DiGraph {
    fn from(graph: CompleteGraph) -> Self {
        let num_nodes = graph.num_nodes;
        let mut matrix = Array2::from_elem((num_nodes, num_nodes), true);
        matrix
            .diag_mut()
            .into_iter()
            .for_each(|entry| *entry = false);

        Self::from_adjacency_matrix(&matrix.view())
    }
}

impl IntoAllPathsIterator for DiGraph {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.edges_list.len()
    }

    #[inline]
    fn all_paths(&self, from: usize, to: usize, depth: usize) -> impl Iterator<Item = Vec<usize>> {
        AllPathsFromDiGraphIter::new(self.clone(), from, to, depth)
    }
}

/// An iterator over all paths in a directed graph.
#[pyclass]
#[derive(Clone, Debug)]
pub struct AllPathsFromDiGraphIter {
    graph: DiGraph,
    to: usize,
    depth: usize,
    stack: Vec<VecDeque<usize>>,
    visited: Vec<usize>,
}

impl AllPathsFromDiGraphIter {
    #[inline]
    pub fn new(graph: DiGraph, from: usize, to: usize, depth: usize) -> Self {
        let stack = vec![graph.get_adjacent_nodes(from).to_vec().into()];
        let visited = vec![from];

        Self {
            graph,
            to,
            depth,
            stack,
            visited,
        }
    }
}

impl Iterator for AllPathsFromDiGraphIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        // The current implementation was derived from
        // the `all_simple_path` function from
        // the petgraph Rust library.

        // Get list of child nodes
        while let Some(children) = self.stack.last_mut() {
            // Get first node in children
            if let Some(child) = children.pop_front() {
                if self.visited.len() < self.depth {
                    if child == self.to && self.visited.len() + 1 == self.depth {
                        let mut path = self.visited.clone();
                        path.push(self.to);
                        return Some(path);
                    } else {
                        self.visited.push(child);
                        self.stack
                            .push(self.graph.get_adjacent_nodes(child).to_vec().into());
                    }
                } else {
                    self.stack.pop();
                    self.visited.pop();
                }
            } else {
                // No more node to visit in children
                self.stack.pop();
                self.visited.pop();
            }
        }

        None
    }
}

impl PathsIterator for AllPathsFromDiGraphIter {
    fn depth(&self) -> Option<usize> {
        Some(self.depth)
    }
}

#[pymethods]
impl DiGraph {
    #[staticmethod]
    fn py_from_adjacency_matrix(adjacency_matrix: PyReadonlyArray2<'_, bool>) -> Self {
        Self::from_adjacency_matrix(&adjacency_matrix.as_array())
    }

    #[staticmethod]
    fn py_from_complete_graph(graph: CompleteGraph) -> Self {
        graph.into()
    }

    /// Insert two additional nodes in the graph:
    ///
    /// a `from` node, that is connected to every other node in the graph;
    /// and a `to` node, where every other node is connected to this node.
    ///
    /// If `direct_path` is `true`, then the `from` node is connected to the
    /// `to` node.
    ///
    /// Return the indices of the two nodes in the graph.
    #[pyo3(signature = (direct_path=true))]
    fn py_insert_from_and_to_nodes(&mut self, direct_path: bool) -> (usize, usize) {
        self.insert_from_and_to_nodes(direct_path)
    }
}

/*
fn all_paths_array_from_graph(graph: usize, depth: u32) -> usize {
    todo!()
}*/

/*
#[pyfunction]
fn all_paths_iter_from_graph(
    graph: Graph,
    from: usize,
    to: usize,
    depth: usize,
) -> AllPathsFromGraphIter {
    AllPathsFromGraphIter::new(graph, from, to, depth)
}

#[pyfunction]
fn all_paths_iter_from_adjacency_matrix(
    graph: Graph,
    from: usize,
    to: usize,
    depth: usize,
) -> AllPathsFromGraphIter {
    AllPathsFromGraphIter::new(graph, from, to, depth)
}
/// Generate all paths of length `depth`.
fn all_paths_array_from_num_nodes(num_nodes: usize, depth: usize) -> usize {
    todo!()
}

fn all_paths_array_from_num_nodes(num_nodes: usize, depth: u32) -> usize {
    todo!()
}*/

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "graph")?;
    m.add_class::<CompleteGraph>()?;
    m.add_class::<DiGraph>()?;
    //m.add_function(wrap_pyfunction!(all_paths_iter_from_graph, m)?)?;

    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cmp::Ordering;

    use ndarray::array;
    use rstest::*;

    fn compare_paths(path1: &[usize], path2: &[usize]) -> Ordering {
        let mut iter1 = path1.iter();
        let mut iter2 = path2.iter();

        loop {
            match (iter1.next(), iter2.next()) {
                (Some(a), Some(b)) => {
                    match a.cmp(b) {
                        Ordering::Equal => continue,
                        ordering => return ordering,
                    }
                },
                (Some(_), None) => return Ordering::Greater,
                (None, Some(_)) => return Ordering::Less,
                (None, None) => return Ordering::Equal,
            }
        }
    }

    #[test]
    fn test_sort_paths() {
        let mut got = [
            [1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0],
        ];

        got.sort_by(|path1, path2| compare_paths(path1, path2));

        let expected = [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 1],
        ];

        assert_eq!(got, expected);
    }

    #[test]
    #[should_panic(expected = "adjacency matrix must be square")]
    fn test_di_graph_from_nonsquare_matrix() {
        let matrix = Array2::default((10, 9));
        let _ = DiGraph::from_adjacency_matrix(&matrix.view());
    }

    #[rstest]
    #[case(1)]
    #[case(10)]
    fn test_di_graph_from_false_matrix(#[case] num_nodes: usize) {
        let matrix = Array2::default((num_nodes, num_nodes));
        let graph = DiGraph::from_adjacency_matrix(&matrix.view());
        assert!(graph.all_paths(0, 0, 0).count() == 0);
    }

    #[rstest]
    #[case(1, 1)]
    #[case(3, 1)]
    #[case(4, 3)]
    #[ignore]
    fn test_complete_vs_di_graph_return_same_paths(#[case] num_nodes: usize, #[case] depth: usize) {
        let from = 0;
        let to = num_nodes - 1;
        let complete_graph = CompleteGraph::new(num_nodes);
        let mut complete_graph_paths: Vec<_> = complete_graph.all_paths(from, to, depth).collect();
        complete_graph_paths.sort_by(|path1, path2| compare_paths(path1, path2));

        let di_graph: DiGraph = complete_graph.into();
        let mut di_graph_paths: Vec<_> = di_graph.all_paths(from, to, depth).collect();
        di_graph_paths.sort_by(|path1, path2| compare_paths(path1, path2));

        assert_eq!(complete_graph_paths, di_graph_paths);
    }

    #[rstest]
    #[case(9, 1, array![[0], [1], [2], [3], [4], [5], [6], [7], [8]])]
    #[case(3, 1, array![[0], [1], [2]])]
    #[case(
        3,
        2,
        array![[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    )]
    #[case(
        3,
        3,
        array![
            [0, 1, 0],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [1, 0, 1],
            [1, 0, 2],
            [1, 2, 0],
            [1, 2, 1],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 2],
        ]
    )]
    fn test_di_graph_all_paths(
        #[case] num_nodes: usize,
        #[case] depth: usize,
        #[case] expected: Array2<usize>,
    ) {
        let mut graph: DiGraph = CompleteGraph::new(num_nodes).into();
        let (from, to) = graph.insert_from_and_to_nodes(true);
        let paths = graph
            .all_paths(from, to, depth + 2)
            .map(|path| path[1..path.len() - 1].to_vec());
        let got = collect_paths_in_array(paths, depth);

        assert_eq!(got, expected);
    }

    #[rstest]
    #[case(9, 2)]
    #[case(3, 3)]
    fn test_di_graph_returns_sorted_paths(#[case] num_nodes: usize, #[case] depth: usize) {
        let mut graph: DiGraph = CompleteGraph::new(num_nodes).into();
        let (from, to) = graph.insert_from_and_to_nodes(true);
        let got: Vec<_> = graph.all_paths(from, to, depth + 2).collect();

        let mut expected = got.clone();

        expected.sort_by(|path1, path2| compare_paths(path1, path2));

        assert_eq!(got, expected);
    }
}

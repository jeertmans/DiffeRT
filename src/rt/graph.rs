use core::ops::Range;
use std::iter::FlatMap;

use numpy::{
    ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView2, Axis},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2,
};
use pyo3::prelude::*;

/// An iterator over paths that have the same depth.
trait PathsIterator: Iterator<Item = Vec<usize>> {
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

struct PathsIter<T> {
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

fn collect_paths_in_array(paths: impl Iterator<Item = Vec<usize>>, depth: usize) -> Array2<usize> {
    let mut flat_vec = Vec::new();
    let mut num_paths = 0;
    paths.for_each(|path| {
        num_paths += 1;
        flat_vec.extend_from_slice(path.as_ref());
    });
    Array2::from_shape_vec((num_paths, depth), flat_vec).unwrap()
}

trait IntoAllPathsIterator {
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
        collect_paths_in_array(self.all_paths, self.num_nodes)
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
struct CompleteGraph {
    /// Number of nodes.
    num_nodes: usize,
}

impl CompleteGraph {
    fn new(num_nodes: usize) -> CompleteGraph {
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
        //AllPathsFromCompleteGraphIter::new(self.num_nodes, from, to, depth)
        todo!()
    }
}

struct AllPathsFromCompleteGraphIter {
    num_nodes: usize,
    from: usize,
    to: usize,
    depth: usize,
    path: Vec<usize>,
    done: bool,
}

impl AllPathsFromCompleteGraphIter {
    #[inline]
    fn new(num_nodes: usize, from: usize, to: usize, depth: usize) -> Self {
        let mut path = vec![0; depth];
        let done = if depth >= 2 {
            path[0] = from;
            path[depth - 1] = to;

            //for i in (1..self.depth - 1) {
            //    path[i] = i - 1;
            //}

            depth == 2 && from == to // Can't generate a 2-depth path
        } else {
            true // Can't generate path of depth < 2
        };

        Self {
            num_nodes,
            from,
            to,
            depth,
            path,
            done,
        }
    }
}

impl Iterator for AllPathsFromCompleteGraphIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        /*

        let mut i = self.depth - 2;

        loop {
            self.path[i] = (self.path[i - 1] + 1) % self.num_nodes;

        }*/

        None
    }
}

impl PathsIterator for AllPathsFromCompleteGraphIter {
    fn depth(&self) -> Option<usize> {
        Some(self.depth)
    }

    /*
    fn as_array(&mut self) -> Array2<usize> {
        let num_nodes = self.num_nodes;
        let depth = self.depth;
        if depth == 0 {
            // One path of size 0
            return Array2::default((0, 1));
        } else if num_nodes == 0 {
            // Zero path of size depth
            return Array2::default((depth, 0));
        } else if depth == 1 {
            let mut paths = Array2::default((1, num_nodes));

            for j in 0..num_nodes {
                paths[(0, j)] = j;
            }
            return paths;
        }
        let depth_u32: u32 = depth.try_into().expect("depth cannot exceed u32's maximum value for collecting as an array");
        let num_choices = num_nodes - 1;
        let num_candidates_per_batch = num_choices.pow(depth_u32 - 1);
        let num_candidates = num_nodes * num_candidates_per_batch;

        let mut paths = Array2::default((depth, num_candidates));
        let mut batch_size = num_candidates_per_batch;
        let mut fill_value = 0;

        for i in 0..depth {
            for j in (0..num_nodes).step_by(batch_size) {
                if i > 0 && fill_value == paths[(i - 1, j)] {
                    fill_value = (fill_value + 1) % num_nodes;
                }

                paths
                    .slice_mut(s![i, j..(j + batch_size)])
                    .fill(fill_value);
                fill_value = (fill_value + 1) % num_nodes;
            }
            batch_size /= num_choices;
        }

        paths
    }*/
}

/// A directed graph.
#[pyclass]
#[derive(Clone, Debug)]
struct DiGraph {
    /// List of list of edges,
    /// where edges[i] is the list of adjacent nodes
    /// of node i.
    edges: Vec<Vec<usize>>,
}

impl DiGraph {
    #[inline]
    fn get_adjacent_nodes(&self, node: usize) -> &[usize] {
        self.edges[node].as_ref()
    }

    fn from_adjacency_matrix(adjacency_matrix: &ArrayView2<bool>) -> Self {
        debug_assert!(
            adjacency_matrix.is_square(),
            "adjacency matrix must be square"
        );
        let edges = adjacency_matrix
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                row.indexed_iter()
                    .filter_map(|(index, &item)| if item { Some(index) } else { None })
                    .collect()
            })
            .collect();

        Self { edges }
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
        self.edges.len()
    }

    #[inline]
    fn all_paths(&self, from: usize, to: usize, depth: usize) -> impl Iterator<Item = Vec<usize>> {
        let mut 
        AllPathsFromDiGraphIter::new(self.clone(), from, to, depth)
    }
}

struct AllPathsFromDiGraphIter {
    graph: DiGraph,
    from: usize,
    to: usize,
    depth: usize,
    stack: Vec<usize>,
}

impl AllPathsFromDiGraphIter {
    #[inline]
    fn new(graph: DiGraph, from: usize, to: usize, depth: usize) -> Self {
        let stack = vec![graph.get_adjacent_nodes(from)];
        
        Self {
            graph,
            from,
            to,
            depth,
            stack,
        }
    }
}

impl Iterator for AllPathsFromDiGraphIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(children) = self.stack.last_mut() {

        }
        let adjacent_nodes = self.graph.get_adjacent_nodes[self.from];
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
    //m.add_function(wrap_pyfunction!(all_paths_iter_from_graph, m)?)?;

    Ok(m)
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;
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
    #[case(0)]
    #[case(1)]
    #[case(10)]
    fn test_di_graph_from_false_matrix(#[case] num_nodes: usize) {
        let matrix = Array2::default((num_nodes, num_nodes));
        let graph = DiGraph::from_adjacency_matrix(&matrix.view());
        assert!(graph.all_paths(0, 0, 1).count() == 0);
    }

    #[rstest]
    #[case(1, 1)]
    #[case(3, 1)]
    #[case(4, 3)]
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
}

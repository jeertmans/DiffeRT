#![allow(unknown_lints, non_local_definitions)]

use std::collections::VecDeque;

use numpy::{
    ndarray::{parallel::prelude::*, Array2, ArrayView2, Axis},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2,
};
use pyo3::{prelude::*, types::PyType};

/// NodeId type.
pub type NodeId = usize;

/// An iterator over paths that have contant depth.
pub trait PathsIterator: Iterator<Item = Vec<NodeId>> {
    /// Hint about the actual path depth.
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
    #[inline]
    fn collect_array(&mut self) -> Array2<NodeId> {
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

    fn into_array_chunks_iter(self, chunk_size: usize) -> PathsChunksIter<Self>
    where
        Self: Sized,
    {
        PathsChunksIter {
            iter: self.into_iter(),
            chunk_size,
        }
    }
}

/// Iterator that collects consecutive
/// paths into array chunks.
#[derive(Clone, Debug)]
pub struct PathsChunksIter<I> {
    iter: I,
    chunk_size: usize,
}

impl<I> Iterator for PathsChunksIter<I>
where
    I: Iterator<Item = Vec<NodeId>>,
{
    type Item = Array2<NodeId>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|mut path| {
            let mut num_paths = 1;
            let depth = path.len();
            for _ in 1..(self.chunk_size) {
                match self.iter.next() {
                    Some(other_path) => {
                        path.extend_from_slice(other_path.as_ref());
                        num_paths += 1;
                    },
                    None => break,
                }
            }
            Array2::from_shape_vec((num_paths, depth), path).unwrap()
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, maybe_upper) = self.iter.size_hint();

        (
            lower.div_ceil(self.chunk_size),
            maybe_upper.map(|upper| upper.div_ceil(self.chunk_size)),
        )
    }
}

pub mod complete {
    use super::*;

    /// A complete graph, i.e.,
    /// a simple undirected graph in which every pair of
    /// distinc nodes is connected by a unique edge.
    #[pyclass]
    #[derive(Clone, Debug)]
    pub struct CompleteGraph {
        /// Number of nodes.
        pub(crate) num_nodes: usize,
    }

    #[pymethods]
    impl CompleteGraph {
        #[new]
        #[inline]
        pub fn new(num_nodes: usize) -> CompleteGraph {
            Self { num_nodes }
        }

        /// Return an iterator over all paths of length ``depth``
        /// from node ``from`` to node ``to``.
        ///
        /// .. note::
        ///
        ///     Unlike for :py:class:`DiGraph`'s iterators, ``from_`` and
        ///     ``to`` nodes do not need to be part of the graph
        ///     (i.e., ``node_id >= num_nodes``).
        ///     This is especially useful to generate all
        ///     paths from ``from_`` to ``to``, where ``from_`` and
        ///     ``to`` will only ever appear in the first and last
        ///     position, respectively.
        ///
        ///     Therefore, those iterators are equivalents:
        ///
        ///     >>> from differt.rt.graph import CompleteGraph, DiGraph
        ///     >>>
        ///     >>> num_nodes, depth = 100, 5
        ///     >>> complete_graph = CompleteGraph(num_nodes)
        ///     >>> di_graph = DiGraph.from_complete_graph(complete_graph)
        ///     >>> from_, to = (
        ///     ...     di_graph.insert_from_and_to_nodes(
        ///     ...         direct_path=True
        ///     ...     )
        ///     ... )
        ///     >>>
        ///     >>> iter1 = complete_graph.all_paths(from_, to, depth)
        ///     >>> iter2 = di_graph.all_paths(from_, to, depth)
        ///     >>> assert(
        ///     ...     all(
        ///     ...         np.array_equal(p1, p2)
        ///     ...         for p1, p2 in zip(iter1, iter2)
        ///     ...     )
        ///     ... )
        ///
        ///     This note also applies to :py:meth:`all_paths_array` and
        ///     :py:meth:`all_paths_array_chunks`.
        ///
        /// Args:
        ///     from_ (int): The node index to find the paths from.
        ///     to (int): The node index to find the paths to.
        ///     depth (int): The number of nodes to include in each path.
        ///     include_from_and_to (bool): Whether to include or not ``from``
        ///         and ``to`` nodes in the output paths. If set to
        ///         :py:data:`False`, the output paths will include
        ///         ``depth - 2`` nodes.
        ///
        /// Return:
        ///     AllPathsFromCompleteGraphIter: An iterator over all paths.
        #[cfg(not(doctest))]
        #[pyo3(signature = (from, to, depth, *, include_from_and_to = true))]
        #[pyo3(text_signature = "(self, from_, to, depth, *, include_from_and_to = True)")]
        pub fn all_paths(
            &self,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
        ) -> AllPathsFromCompleteGraphIter {
            AllPathsFromCompleteGraphIter::new(self.clone(), from, to, depth, include_from_and_to)
        }

        /// Return an array of all paths of length ``depth``
        /// from node ``from`` to node ``to``.
        ///
        /// Args:
        ///     from_ (int): The node index to find the paths from.
        ///     to (int): The node index to find the paths to.
        ///     depth (int): The number of nodes to include in each path.
        ///     include_from_and_to (bool): Whether to include or not ``from``
        ///         and ``to`` nodes in the output paths. If set to
        ///         :py:data:`False`, the output paths will include
        ///         ``depth - 2`` nodes.
        ///
        /// Return:
        ///     ``UInt[ndarray, "num_paths path_depth"]``:
        ///         An array of all paths.
        #[pyo3(signature = (from, to, depth, *, include_from_and_to = true))]
        #[pyo3(text_signature = "(self, from_, to, depth, *, include_from_and_to = True)")]
        fn all_paths_array<'py>(
            &self,
            py: Python<'py>,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
        ) -> &'py PyArray2<NodeId> {
            AllPathsFromCompleteGraphIter::new(self.clone(), from, to, depth, include_from_and_to)
                .collect_array()
                .into_pyarray(py)
        }

        /// Return an iterator over all paths of length ``depth``
        /// from node ``from`` to node ``to``, grouped in chunks of
        /// size of max. ``chunk_size``.
        ///
        /// Args:
        ///     from_ (int): The node index to find the paths from.
        ///     to (int): The node index to find the paths to.
        ///     depth (int): The number of nodes to include in each path.
        ///     include_from_and_to (bool): Whether to include or not ``from``
        ///         and ``to`` nodes in the output paths. If set to
        ///         :py:data:`False`, the output paths will include
        ///         ``depth - 2`` nodes.
        ///     chunk_size (int): The size of each chunk.
        ///
        /// Return:
        ///     AllPathsFromCompleteGraphChunksIter:
        ///         An iterator over all paths, as array chunks.
        #[pyo3(signature = (from, to, depth, *, include_from_and_to = true, chunk_size = 1000))]
        #[pyo3(
            text_signature = "(self, from_, to, depth, *, include_from_and_to = True, chunk_size \
                              = 1000)"
        )]
        pub fn all_paths_array_chunks(
            &self,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
            chunk_size: usize,
        ) -> AllPathsFromCompleteGraphChunksIter {
            assert!(chunk_size > 0, "chunk size must be strictly positive");
            AllPathsFromCompleteGraphChunksIter {
                iter: AllPathsFromCompleteGraphIter::new(
                    self.clone(),
                    from,
                    to,
                    depth,
                    include_from_and_to,
                )
                .into_array_chunks_iter(chunk_size),
            }
        }
    }

    /// An iterator over all paths in a complete graph.
    ///
    /// Note:
    ///     Even though this iterator is generally sized, this is not true
    ///     when its length is so large that overflow occured when computing
    ///     its theoritical length.
    ///     For lengths close or above ``usize::MAX``, do not rely
    ///     on the provided size hint nor the length.
    #[pyclass]
    #[derive(Clone, Debug)]
    pub struct AllPathsFromCompleteGraphIter {
        graph: CompleteGraph,
        to: NodeId,
        depth: usize,
        include_from_and_to: bool,
        visited: Vec<NodeId>,
        counter: Vec<usize>,
        paths_count: usize,
        paths_total_count: usize,
    }

    impl AllPathsFromCompleteGraphIter {
        #[inline]
        pub fn new(
            graph: CompleteGraph,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
        ) -> Self {
            use std::cmp::Ordering::*;

            let mut visited = Vec::with_capacity(depth);
            let mut counter = Vec::with_capacity(depth);
            visited.push(from);
            counter.push(graph.num_nodes); // num_nodes means we visited all
            // first nodes, because first not is fixed

            let paths_count = 0;
            let paths_total_count = match depth.cmp(&2) {
                Less => 0,
                Equal if from == to => 0,
                Equal => 1,
                Greater => {
                    let num_intermediate_nodes = (depth - 2) as u32;
                    let num_nodes = graph.num_nodes;

                    let from_in_graph = from < num_nodes;
                    let to_in_graph = to < num_nodes;

                    match (from_in_graph, to_in_graph) {
                        (true, true) => {
                            //This solution was obtained thanks to user @ronno, see:
                            // https://math.stackexchange.com/a/4874894/1297520.
                            let depth_minus_1 = depth.saturating_sub(1) as u32;
                            let depth_minus_2 = depth.saturating_sub(2) as u32;
                            let num_nodes_minus_1 = num_nodes.saturating_sub(1);
                            if from != to {
                                num_nodes_minus_1
                                    .saturating_pow(depth_minus_1)
                                    .saturating_add_signed(if depth_minus_1 % 2 == 0 {
                                        -1
                                    } else {
                                        1
                                    })
                                    / num_nodes
                            } else {
                                num_nodes_minus_1
                                    .saturating_pow(depth_minus_2)
                                    .saturating_add_signed(if depth_minus_2 % 2 == 0 {
                                        -1
                                    } else {
                                        1
                                    })
                                    * num_nodes_minus_1
                                    / num_nodes
                            }
                        },
                        (false, false) => {
                            // (num_nodes) * (num_nodes - 1)^(num_intermediate_nodes - 1)
                            (num_nodes).saturating_mul(
                                num_nodes
                                    .saturating_sub(1)
                                    .saturating_pow(num_intermediate_nodes.saturating_sub(1)),
                            )
                        },
                        _ => {
                            // (num_nodes - 1)^num_intermediate_nodes
                            (num_nodes.saturating_sub(1)).saturating_pow(num_intermediate_nodes)
                        },
                    }
                },
            };

            Self {
                graph,
                to,
                depth,
                include_from_and_to,
                visited,
                counter,
                paths_count,
                paths_total_count,
            }
        }
    }

    impl Iterator for AllPathsFromCompleteGraphIter {
        type Item = Vec<NodeId>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            // This is a 'specialized' implementation
            // of AllPathsFromDiGraphIter, optimized
            // for complete graphs.

            while !self.visited.is_empty() {
                let prev_depth = self.visited.len(); // Can't be 0
                let prev_node_id = self.visited[prev_depth - 1];

                // Is it the last node we should add?
                if prev_depth + 1 == self.depth {
                    // Check if not a cycle
                    let maybe_path = if prev_node_id != self.to {
                        let path = if self.include_from_and_to {
                            let mut path = self.visited.clone();
                            path.push(self.to);
                            path
                        } else {
                            self.visited[1..].to_vec()
                        };
                        self.paths_count += 1;
                        Some(path)
                    } else {
                        None
                    };

                    // Update visited nodes
                    let mut index = prev_depth - 1;
                    loop {
                        if index == 0 {
                            self.visited.pop();
                            return maybe_path;
                        }

                        let prev_count = self.counter[index];

                        if prev_count < self.graph.num_nodes {
                            self.visited[index] = (self.visited[index] + 1) % self.graph.num_nodes;
                            self.counter[index] += 1;

                            if self.visited[index] != self.visited[index - 1] {
                                break;
                            };
                        } else {
                            self.visited.pop();
                            self.counter.pop();
                            index -= 1;
                        }
                    }

                    if maybe_path.is_some() {
                        return maybe_path;
                    }
                } else {
                    // Otherwise, we should visit one more node
                    let (next_node_id, count) = if prev_node_id != 0 {
                        (0, 1) // First node index is zero, we visited 1 node
                    } else {
                        (1, 2) // First node index is one, we alreadi visited 2 nodes
                        // because the first one was skipped
                    };
                    self.visited.push(next_node_id);
                    self.counter.push(count);
                }
            }
            None
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let size = self.paths_total_count.saturating_sub(self.paths_count);

            (size, Some(size))
        }
    }

    impl ExactSizeIterator for AllPathsFromCompleteGraphIter {}

    impl PathsIterator for AllPathsFromCompleteGraphIter {
        #[inline]
        fn depth(&self) -> Option<usize> {
            if self.include_from_and_to {
                Some(self.depth)
            } else {
                Some(self.depth.saturating_sub(2))
            }
        }
    }

    #[pymethods]
    impl AllPathsFromCompleteGraphIter {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__<'py>(
            mut slf: PyRefMut<'py, Self>,
            py: Python<'py>,
        ) -> Option<&'py PyArray1<NodeId>> {
            slf.next().map(|path| PyArray1::from_vec(py, path))
        }

        fn __len__(&self) -> usize {
            self.len()
        }
    }

    /// An iterator over all paths in a complete graph,
    /// as array chunks.
    #[pyclass]
    #[derive(Clone, Debug)]
    pub struct AllPathsFromCompleteGraphChunksIter {
        iter: PathsChunksIter<AllPathsFromCompleteGraphIter>,
    }

    impl Iterator for AllPathsFromCompleteGraphChunksIter {
        type Item = Array2<NodeId>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.iter.size_hint()
        }
    }

    impl ExactSizeIterator for AllPathsFromCompleteGraphChunksIter {}

    #[pymethods]
    impl AllPathsFromCompleteGraphChunksIter {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__<'py>(
            mut slf: PyRefMut<'py, Self>,
            py: Python<'py>,
        ) -> Option<&'py PyArray2<NodeId>> {
            slf.iter.next().map(|paths| paths.into_pyarray(py))
        }

        fn __len__(&self) -> usize {
            self.len()
        }
    }
}

pub mod directed {
    use super::{complete::CompleteGraph, *};

    /// A directed graph.
    #[pyclass]
    #[derive(Clone, Debug)]
    pub struct DiGraph {
        /// List of list of edges,
        /// where edges[i] is the list of adjacent nodes
        /// of node i.
        edges_list: Vec<Vec<NodeId>>,
    }

    impl DiGraph {
        #[inline]
        pub fn get_adjacent_nodes(&self, node: NodeId) -> &[NodeId] {
            self.edges_list[node].as_ref()
        }

        #[inline]
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
    }

    #[pymethods]
    impl DiGraph {
        /// Create a directed graph from an adjacency matrix.
        ///
        /// Each row of the adjacency matrix ``M`` contains boolean
        /// entries: if ``M[i, j]`` is :py:data:`True`, then node ``i`` is
        /// connected to node ``j``.
        ///
        /// Args:
        ///     adjacency_matrix (``Bool[ndarray, "num_nodes num_nodes"]``):
        ///         The adjacency matrix.
        ///
        /// Return:
        ///     DiGraph: A directed graph.
        #[classmethod]
        #[pyo3(name = "from_adjacency_matrix")]
        #[pyo3(text_signature = "(cls, adjacency_matrix)")]
        fn py_from_adjacency_matrix(
            _cls: &PyType,
            adjacency_matrix: PyReadonlyArray2<'_, bool>,
        ) -> Self {
            Self::from_adjacency_matrix(&adjacency_matrix.as_array())
        }

        /// Create a directed graph from a complete graph.
        ///
        /// This is equivalent to creating a directed graph from
        /// an adjacency matrix will all entries equal to :py:data:`True`,
        /// except on the main diagonal (i.e., no loop).
        ///
        /// Args:
        ///     graph (CompleteGraph): The complete graph.
        ///
        /// Return:
        ///     DiGraph: A directed graph.
        #[classmethod]
        #[pyo3(name = "from_complete_graph")]
        #[pyo3(text_signature = "(cls, graph)")]
        fn py_from_complete_graph(_cls: &PyType, graph: CompleteGraph) -> Self {
            graph.into()
        }

        /// Insert two additional nodes in the graph:
        ///
        /// - a ``from`` node, that is connected to every other node in the
        ///   graph;
        /// - and a ``to`` node, where every other node is connected to this
        ///   node.
        ///
        /// If ``direct_path`` is :py:data:`True`, then the ``from`` node is
        /// connected to the ``to`` node.
        ///
        /// Args:
        ///     direct_path (bool): Whether to create a direction connection
        ///         between ``from`` and ``to`` nodes.
        ///
        /// Return:
        ///     tuple[int, int]:
        ///         The indices of the two added nodes in the graph.
        #[pyo3(signature = (*, direct_path=true))]
        #[pyo3(text_signature = "(self, *, direct_path=True)")]
        pub fn insert_from_and_to_nodes(&mut self, direct_path: bool) -> (NodeId, NodeId) {
            let from = self.edges_list.len();
            let to = from + 1;

            // Every node is connected to `to`.
            self.edges_list.iter_mut().for_each(|edges| edges.push(to));

            // `from` is connected to every node except itself
            let mut from_edges: Vec<NodeId> = (0..from).collect();

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

        /// Return an iterator over all paths of length ``depth``
        /// from node ``from`` to node ``to``.
        ///
        /// Args:
        ///     from_ (int): The node index to find the paths from.
        ///     to (int): The node index to find the paths to.
        ///     depth (int): The number of nodes to include in each path.
        ///     include_from_and_to (bool): Whether to include or not ``from``
        ///         and ``to`` nodes in the output paths. If set to
        ///         :py:data:`False`, the output paths will include
        ///         ``depth - 2`` nodes.
        ///
        /// Return:
        ///     AllPathsFromDiGraphIter: An iterator over all paths.
        #[pyo3(signature = (from, to, depth, *, include_from_and_to = true))]
        #[pyo3(text_signature = "(self, from_, to, depth, *, include_from_and_to = True)")]
        pub fn all_paths(
            &self,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
        ) -> AllPathsFromDiGraphIter {
            AllPathsFromDiGraphIter::new(self.clone(), from, to, depth, include_from_and_to)
        }

        /// Return an array of all paths of length ``depth``
        /// from node ``from`` to node ``to``.
        ///
        /// Args:
        ///     from_ (int): The node index to find the paths from.
        ///     to (int): The node index to find the paths to.
        ///     depth (int): The number of nodes to include in each path.
        ///     include_from_and_to (bool): Whether to include or not ``from``
        ///         and ``to`` nodes in the output paths. If set to
        ///         :py:data:`False`, the output paths will include
        ///         ``depth - 2`` nodes.
        ///
        /// Return:
        ///     ``UInt[ndarray, "num_paths path_depth"]``:
        ///         An array of all paths.
        #[pyo3(signature = (from, to, depth, *, include_from_and_to = true))]
        #[pyo3(text_signature = "(self, from_, to, depth, *, include_from_and_to = True)")]
        fn all_paths_array<'py>(
            &self,
            py: Python<'py>,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
        ) -> &'py PyArray2<NodeId> {
            AllPathsFromDiGraphIter::new(self.clone(), from, to, depth, include_from_and_to)
                .collect_array()
                .into_pyarray(py)
        }

        /// Return an iterator over all paths of length ``depth``
        /// from node ``from`` to node ``to``, grouped in chunks of
        /// size of max. ``chunk_size``.
        ///
        /// Args:
        ///     from_ (int): The node index to find the paths from.
        ///     to (int): The node index to find the paths to.
        ///     depth (int): The number of nodes to include in each path.
        ///     include_from_and_to (bool): Whether to include or not ``from``
        ///         and ``to`` nodes in the output paths. If set to
        ///         :py:data:`False`, the output paths will include
        ///         ``depth - 2`` nodes.
        ///     chunk_size (int): The size of each chunk.
        ///
        /// Return:
        ///     AllPathsFromDiGraphChunksIter:
        ///         An iterator over all paths, as array chunks.
        #[pyo3(signature = (from, to, depth, *, include_from_and_to = true, chunk_size = 1000))]
        #[pyo3(
            text_signature = "(self, from_, to, depth, *, include_from_and_to = True, chunk_size \
                              = 1000)"
        )]
        pub fn all_paths_array_chunks(
            &self,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
            chunk_size: usize,
        ) -> AllPathsFromDiGraphChunksIter {
            assert!(chunk_size > 0, "chunk size must be strictly positive");
            AllPathsFromDiGraphChunksIter {
                iter: AllPathsFromDiGraphIter::new(
                    self.clone(),
                    from,
                    to,
                    depth,
                    include_from_and_to,
                )
                .into_array_chunks_iter(chunk_size),
            }
        }
    }

    impl From<CompleteGraph> for DiGraph {
        #[inline]
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

    /// An iterator over all paths in a directed graph.
    #[pyclass]
    #[derive(Clone, Debug)]
    pub struct AllPathsFromDiGraphIter {
        graph: DiGraph,
        to: NodeId,
        depth: usize,
        include_from_and_to: bool,
        stack: Vec<VecDeque<NodeId>>,
        visited: Vec<NodeId>,
    }

    impl AllPathsFromDiGraphIter {
        #[inline]
        pub fn new(
            graph: DiGraph,
            from: NodeId,
            to: NodeId,
            depth: usize,
            include_from_and_to: bool,
        ) -> Self {
            let mut stack = Vec::with_capacity(depth);
            let mut visited = Vec::with_capacity(depth);
            stack.push(graph.get_adjacent_nodes(from).to_vec().into());
            visited.push(from);

            Self {
                graph,
                to,
                depth,
                include_from_and_to,
                stack,
                visited,
            }
        }
    }

    impl Iterator for AllPathsFromDiGraphIter {
        type Item = Vec<NodeId>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            // The current implementation was derived from
            // the `all_simple_path` function from
            // the petgraph Rust library.

            // While we have nodes to visit
            while let Some(children) = self.stack.last_mut() {
                // Is it the last node we should add?
                if self.visited.len() + 1 == self.depth {
                    // If we can actually reach `to`
                    if children.binary_search(&self.to).is_ok() {
                        let path = if self.include_from_and_to {
                            let mut path = self.visited.clone();
                            path.push(self.to);
                            path
                        } else {
                            self.visited[1..].to_vec()
                        };
                        self.stack.pop();
                        self.visited.pop();
                        return Some(path);
                    }
                    // Otherwise, we quit the current node (second-last).
                    else {
                        self.stack.pop();
                        self.visited.pop();
                    }
                }
                // Else, try to visit one node deeper
                else if let Some(child) = children.pop_front() {
                    self.visited.push(child);
                    self.stack
                        .push(self.graph.get_adjacent_nodes(child).to_vec().into());
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
        #[inline]
        fn depth(&self) -> Option<usize> {
            if self.include_from_and_to {
                Some(self.depth)
            } else {
                Some(self.depth.saturating_sub(2))
            }
        }
    }

    #[pymethods]
    impl AllPathsFromDiGraphIter {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__<'py>(
            mut slf: PyRefMut<'py, Self>,
            py: Python<'py>,
        ) -> Option<&'py PyArray1<NodeId>> {
            slf.next().map(|path| PyArray1::from_vec(py, path))
        }
    }

    /// An iterator over all paths in a directed graph,
    /// as array chunks.
    #[pyclass]
    #[derive(Clone, Debug)]
    pub struct AllPathsFromDiGraphChunksIter {
        iter: PathsChunksIter<AllPathsFromDiGraphIter>,
    }

    impl Iterator for AllPathsFromDiGraphChunksIter {
        type Item = Array2<NodeId>;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next()
        }
    }

    #[pymethods]
    impl AllPathsFromDiGraphChunksIter {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__<'py>(
            mut slf: PyRefMut<'py, Self>,
            py: Python<'py>,
        ) -> Option<&'py PyArray2<NodeId>> {
            slf.iter.next().map(|paths| paths.into_pyarray(py))
        }
    }
}

pub(crate) fn create_module(py: Python<'_>) -> PyResult<&PyModule> {
    let m = pyo3::prelude::PyModule::new(py, "graph")?;
    m.add_class::<complete::CompleteGraph>()?;
    m.add_class::<directed::DiGraph>()?;
    m.add_class::<complete::AllPathsFromCompleteGraphIter>()?;
    m.add_class::<complete::AllPathsFromCompleteGraphChunksIter>()?;
    m.add_class::<directed::AllPathsFromDiGraphIter>()?;
    m.add_class::<directed::AllPathsFromDiGraphChunksIter>()?;

    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::{complete::CompleteGraph, directed::DiGraph, *};

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

    #[rstest]
    #[case(0, 2, 0, 1)] // One path of depth 2 [0, 1]
    #[case(0, 2, 0, 0)] // No path of depth 2 (because can't be [0, 0])
    #[case(4, 2, 0, 1)] // One path of depth 2 [0, 1]
    #[case(1, 3, 1, 2)] // One path of depth 3 [1, 0, 2]
    #[case(8, 5, 8, 9)]
    #[case(8, 5, 0, 9)]
    #[case(8, 5, 8, 0)]
    #[case(4, 4, 4, 0)]
    #[case(4, 3, 0, 0)]
    #[case(4, 3, 0, 1)]
    #[case(4, 4, 0, 1)]
    #[case(4, 4, 0, 1)]
    #[case(6, 4, 0, 1)]
    #[case(4, 5, 0, 1)]
    #[case(4, 5, 0, 0)]
    #[case(3, 5, 0, 1)]
    #[case(8, 5, 0, 1)]
    #[case(5, 3, 8, 9)]
    #[case(5, 3, 0, 9)]
    #[case(5, 3, 8, 0)]
    #[case(5, 3, 0, 1)]
    #[case(4, 2, 0, 1)]
    #[case(4, 3, 0, 1)]
    #[case(4, 4, 0, 1)]
    #[case(4, 5, 0, 1)]
    #[case(4, 6, 0, 1)]
    #[case(4, 7, 0, 1)]
    #[case(4000, 3, 0, 1)]
    fn test_complete_graph_all_paths_len(
        #[case] num_nodes: usize,
        #[case] depth: usize,
        #[case] from: usize,
        #[case] to: usize,
    ) {
        let iter = CompleteGraph::new(num_nodes).all_paths(from, to, depth, false);
        let iter_cloned = iter.clone();
        let got = iter.len();
        let expected = iter.count();

        assert_eq!(got, expected);

        let iter_skipped = iter_cloned.skip(10);

        let got = iter_skipped.len();
        let expected = iter_skipped.count();

        assert_eq!(got, expected);

        let iter_chunks =
            CompleteGraph::new(num_nodes).all_paths_array_chunks(from, to, depth, false, 100);
        let got = iter_chunks.len();
        let expected = iter_chunks.count();

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
        assert!(graph.all_paths(0, 0, 0, true).count() == 0);
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
        #[case] expected: Array2<NodeId>,
    ) {
        let mut graph: DiGraph = CompleteGraph::new(num_nodes).into();
        let (from, to) = graph.insert_from_and_to_nodes(true);
        let got = graph.all_paths(from, to, depth + 2, false).collect_array();

        assert_eq!(got, expected);
    }

    #[rstest]
    #[case(9, 2)]
    #[case(3, 3)]
    fn test_di_graph_returns_sorted_paths(#[case] num_nodes: usize, #[case] depth: usize) {
        let mut graph: DiGraph = CompleteGraph::new(num_nodes).into();
        let (from, to) = graph.insert_from_and_to_nodes(true);
        let got: Vec<_> = graph.all_paths(from, to, depth + 2, true).collect();

        let mut expected = got.clone();

        expected.sort_by(|path1, path2| compare_paths(path1, path2));

        assert_eq!(got, expected);
    }

    #[rstest]
    #[case(5, 3)]
    #[case(8, 3)]
    fn test_di_graph_array_chunks_is_iter(#[case] num_nodes: usize, #[case] depth: usize) {
        let mut graph: DiGraph = CompleteGraph::new(num_nodes).into();
        let (from, to) = graph.insert_from_and_to_nodes(true);
        let iter = graph.all_paths(from, to, depth + 2, true);
        let chunks_iter = graph.all_paths_array_chunks(from, to, depth + 2, true, 1);

        assert!(iter.eq(chunks_iter.map(|chunk_array| chunk_array.as_slice().unwrap().to_vec())));
    }

    #[rstest]
    #[case(5, 3)]
    #[case(8, 3)]
    fn test_complete_graph_and_di_graph_on_equivalent_cases(
        #[case] num_nodes: usize,
        #[case] depth: usize,
    ) {
        let complete_graph = CompleteGraph::new(num_nodes);
        let mut di_graph: DiGraph = complete_graph.clone().into();
        let (from, to) = di_graph.insert_from_and_to_nodes(true);
        let complete_iter = complete_graph.all_paths(from, to, depth + 2, true);
        let di_iter = di_graph.all_paths(from, to, depth + 2, true);

        assert!(complete_iter.eq(di_iter));
    }
}

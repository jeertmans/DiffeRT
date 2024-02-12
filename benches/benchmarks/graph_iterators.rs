use criterion::{black_box, criterion_group, Criterion};

use differt::rt::graph::{CompleteGraph, DiGraph, IntoAllPathsIterator};

fn di_graph_from_complete_graph_all_paths(c: &mut Criterion) {

    let mut graph: DiGraph = CompleteGraph::new(1000).into();
    let (from, to) = graph.insert_from_and_to_nodes(true);

    let mut iter = graph.all_paths(from, to, 5, false);

    c.bench_function("di_graph_from_complete_graph_all_paths", || black_box(iter.next()));
}


criterion_group!(benches, di_graph_from_complete_graph_all_paths);

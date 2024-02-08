#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use differt::rt::graph::{CompleteGraph, DiGraph, IntoAllPathsIterator};

fn di_graph_count_all_paths(num_nodes: usize, depth: usize) -> usize {
    let mut graph: DiGraph = CompleteGraph::new(num_nodes).into();
    let (from, to) = graph.insert_from_and_to_nodes(true);
    graph.all_paths(from, to, depth + 2).count()
}

#[bench]
fn bench_rt_di_graph_all_paths(bencher: &mut Bencher) {
    bencher.iter(|| di_graph_count_all_paths(black_box(10), black_box(3)));
}

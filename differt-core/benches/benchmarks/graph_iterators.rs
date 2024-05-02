use criterion::{black_box, criterion_group, Criterion, Throughput};
use differt_core::rt::graph::{complete::CompleteGraph, directed::DiGraph};

const NUM_NODES: usize = 1000;
const DIRECT_PATH: bool = true;
const DEPTH: usize = 5;
const INCLUDE_FROM_AND_TO: bool = false;

fn complete_graph_all_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_graph_all_paths");
    group.throughput(Throughput::Elements(1));
    let graph = CompleteGraph::new(NUM_NODES);
    let from = NUM_NODES;
    let to = from + 1;

    let mut iter = graph
        .all_paths(from, to, DEPTH, INCLUDE_FROM_AND_TO)
        .cycle();

    group.bench_function("iter", |b| b.iter(|| black_box(iter.next())));

    group.finish();
}

fn complete_graph_all_paths_array_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_graph_all_paths_array_chunks");
    let graph = CompleteGraph::new(NUM_NODES);
    let from = NUM_NODES;
    let to = from + 1;

    for chunk_size in [1, 10, 100, 1000] {
        group.throughput(Throughput::Elements(chunk_size as u64));
        let mut iter = graph
            .all_paths_array_chunks(from, to, DEPTH, INCLUDE_FROM_AND_TO, chunk_size)
            .cycle();

        group.bench_function(format!("{chunk_size}"), |b| {
            b.iter(|| black_box(iter.next()))
        });
    }

    group.finish()
}

fn di_graph_from_complete_graph_all_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("di_graph_from_complete_graph_all_paths");
    group.throughput(Throughput::Elements(1));
    let mut graph: DiGraph = CompleteGraph::new(NUM_NODES).into();
    let (from, to) = graph.insert_from_and_to_nodes(DIRECT_PATH);

    let mut iter = graph
        .all_paths(from, to, DEPTH, INCLUDE_FROM_AND_TO)
        .cycle();

    group.bench_function("iter", |b| b.iter(|| black_box(iter.next())));

    group.finish();
}

fn di_graph_from_complete_graph_all_paths_array_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("di_graph_from_complete_graph_all_paths_array_chunks");
    let mut graph: DiGraph = CompleteGraph::new(NUM_NODES).into();
    let (from, to) = graph.insert_from_and_to_nodes(DIRECT_PATH);

    for chunk_size in [1, 10, 100, 1000] {
        group.throughput(Throughput::Elements(chunk_size as u64));
        let mut iter = graph
            .all_paths_array_chunks(from, to, DEPTH, INCLUDE_FROM_AND_TO, chunk_size)
            .cycle();

        group.bench_function(format!("{chunk_size}"), |b| {
            b.iter(|| black_box(iter.next()))
        });
    }

    group.finish()
}

criterion_group!(
    benches,
    complete_graph_all_paths,
    complete_graph_all_paths_array_chunks,
    di_graph_from_complete_graph_all_paths,
    di_graph_from_complete_graph_all_paths_array_chunks
);

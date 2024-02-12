use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    benchmarks::graph_iterators::benches,
}

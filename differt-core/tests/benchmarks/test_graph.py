"""
This test module tries to provide the same benchmark
as in ./benches/, but in Python, to estimate the overhead
between Rust and Python code, but also to provide meaningful
runs for PGO optimization.
"""

from itertools import cycle

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from differt_core.rt.graph import CompleteGraph, DiGraph

NUM_NODES: int = 1000
DIRECT_PATH: bool = True
DEPTH: int = 5
INCLUDE_FROM_AND_TO: bool = False


@pytest.mark.benchmark(group="complete_graph_all_paths")
def test_complete_graph_all_paths(benchmark: BenchmarkFixture) -> None:
    graph = CompleteGraph(NUM_NODES)
    from_ = NUM_NODES
    to = from_ + 1
    it = cycle(
        graph.all_paths(from_, to, DEPTH, include_from_and_to=INCLUDE_FROM_AND_TO)
    )

    _ = benchmark(lambda: next(it))


@pytest.mark.parametrize("chunk_size", [1, 10, 100, 1000])
@pytest.mark.benchmark(group="complete_graph_all_paths_array_chunks")
def test_complete_graph_all_paths_array_chunks(
    chunk_size: int, benchmark: BenchmarkFixture
) -> None:
    graph = CompleteGraph(NUM_NODES)
    from_ = NUM_NODES
    to = from_ + 1
    it = cycle(
        graph.all_paths_array_chunks(
            from_,
            to,
            DEPTH,
            include_from_and_to=INCLUDE_FROM_AND_TO,
            chunk_size=chunk_size,
        )
    )

    benchmark.extra_info["scale"] = chunk_size
    _ = benchmark(lambda: next(it))


@pytest.mark.benchmark(group="di_graph_from_complete_graph_all_paths")
def test_di_graph_from_complete_graph_all_paths(benchmark: BenchmarkFixture) -> None:
    graph = DiGraph.from_complete_graph(CompleteGraph(NUM_NODES))
    from_, to = graph.insert_from_and_to_nodes(direct_path=DIRECT_PATH)
    it = cycle(
        graph.all_paths(from_, to, DEPTH, include_from_and_to=INCLUDE_FROM_AND_TO)
    )

    _ = benchmark(lambda: next(it))


@pytest.mark.parametrize("chunk_size", [1, 10, 100, 1000])
@pytest.mark.benchmark(group="di_graph_complete_graph_all_paths_array_chunks")
def test_di_graph_from_complete_graph_all_paths_array_chunks(
    chunk_size: int, benchmark: BenchmarkFixture
) -> None:
    graph = DiGraph.from_complete_graph(CompleteGraph(NUM_NODES))
    from_, to = graph.insert_from_and_to_nodes(direct_path=DIRECT_PATH)
    it = cycle(
        graph.all_paths_array_chunks(
            from_,
            to,
            DEPTH,
            include_from_and_to=INCLUDE_FROM_AND_TO,
            chunk_size=chunk_size,
        )
    )

    benchmark.extra_info["scale"] = chunk_size
    _ = benchmark(lambda: next(it))

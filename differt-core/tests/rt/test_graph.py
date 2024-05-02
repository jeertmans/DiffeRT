import pytest

from differt_core.rt.graph import (
    CompleteGraph,
    DiGraph,
)


class TestDiGraph:
    def test_insert_from_and_to_nodes(self) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(5))
        from_, to = graph.insert_from_and_to_nodes()
        assert from_ == 5
        assert to == 6
        from_, to = graph.insert_from_and_to_nodes(direct_path=True)
        assert from_ == 7
        assert to == 8
        from_, to = graph.insert_from_and_to_nodes(direct_path=False)
        assert from_ == 9
        assert to == 10

        with pytest.raises(
            TypeError, match="takes 0 positional arguments but 1 was given"
        ):
            _ = graph.insert_from_and_to_nodes(True)  # type: ignore

    def test_from_graph(self) -> None:
        graph = CompleteGraph(10)
        assert isinstance(graph, CompleteGraph)
        graph = DiGraph.from_complete_graph(graph)
        assert isinstance(graph, DiGraph)

    def test_all_keyword_only_parameters(self) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(5))

        with pytest.raises(
            TypeError, match="takes 3 positional arguments but 4 were given"
        ):
            _ = graph.all_paths(0, 1, 0, True)  # type: ignore

    @pytest.mark.parametrize(
        "num_nodes,depth",
        [
            (10, 1),
            (50, 2),
            (10, 3),
        ],
    )
    def test_all_paths_count_from_complete_graph(
        self, num_nodes: int, depth: int
    ) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(num_nodes))
        from_, to = graph.insert_from_and_to_nodes()
        paths = graph.all_paths(from_, to, depth + 2, include_from_and_to=False)
        num_paths = sum(1 for _ in paths)
        assert num_paths == num_nodes * (num_nodes - 1) ** (depth - 1)
        array = graph.all_paths_array(from_, to, depth + 2, include_from_and_to=False)
        assert array.shape == (num_paths, depth)

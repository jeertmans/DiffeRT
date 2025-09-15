import logging
import re

import numpy as np
import pytest

from differt_core.rt._graph import (
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
            TypeError,
            match="takes 0 positional arguments but 1 was given",
        ):
            _ = graph.insert_from_and_to_nodes(True)  # noqa: FBT003

    @pytest.mark.parametrize("fast_mode", [True, False])
    def test_disconnect_nodes(self, fast_mode: bool) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(6))
        from_, to = graph.insert_from_and_to_nodes()

        nodes = (0, 1, 2, 5)
        graph.disconnect_nodes(*nodes, fast_mode=fast_mode)

        for depth in range(3):
            for path in graph.all_paths(
                from_,
                to,
                depth + 2,
                include_from_and_to=False,
            ):
                for node in nodes:
                    assert node not in path

    @pytest.mark.parametrize("fast_mode", [True, False])
    def test_filter_by_mask(self, fast_mode: bool) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(8))
        from_, to = graph.insert_from_and_to_nodes()

        # Create a mask that disconnects nodes 1, 3, 5, 7 (keeps 0, 2, 4, 6)
        mask = np.array([True, False, True, False, True, False, True, False])
        disconnected_nodes = [1, 3, 5, 7]

        # Count paths before filtering
        paths_before = len(
            list(graph.all_paths(from_, to, 3, include_from_and_to=False))
        )

        # Apply the mask filter
        graph.filter_by_mask(mask, fast_mode=fast_mode)

        # Count paths after filtering and verify no disconnected nodes appear
        paths_after = []
        for path in graph.all_paths(from_, to, 3, include_from_and_to=False):
            paths_after.append(path)
            for node in disconnected_nodes:
                assert node not in path, (
                    f"Path {path} should not contain disconnected node {node}"
                )

        # Verify that we have fewer paths after filtering
        assert len(paths_after) < paths_before, (
            "Should have fewer paths after filtering by mask"
        )

    @pytest.mark.parametrize("fast_mode", [True, False])
    def test_filter_by_mask_all_disconnected(self, fast_mode: bool) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(4))
        from_, to = graph.insert_from_and_to_nodes(direct_path=False)

        # Disconnect all intermediate nodes
        mask = np.array([False, False, False, False])
        graph.filter_by_mask(mask, fast_mode=fast_mode)

        # Should have no paths when all intermediate nodes are disconnected
        paths = list(graph.all_paths(from_, to, 4, include_from_and_to=False))
        assert len(paths) == 0, "Should have no paths when all nodes are disconnected"

    def test_filter_by_mask_wrong_size(self) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(5))

        # Test with mask of wrong size
        wrong_mask = np.array([True, False, True])  # Only 3 elements for 5 nodes
        # Ok to use a smaller mask
        graph.filter_by_mask(wrong_mask, fast_mode=True)

        graph = DiGraph.from_complete_graph(CompleteGraph(5))

        wrong_mask = np.array([
            True,
            False,
            True,
            False,
            False,
            False,
        ])  # 6 elements for 5 nodes
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'mask' length (6) must be smaller than or equal to the number of nodes in the graph (5)"
            ),
        ):
            graph.filter_by_mask(wrong_mask, fast_mode=True)

    @pytest.mark.parametrize("fast_mode", [True, False])
    def test_disconnect_nodes_equivalence(self, fast_mode: bool) -> None:
        complete_graph = CompleteGraph(3)
        di_graph = DiGraph.from_complete_graph(CompleteGraph(6))
        from_, to = di_graph.insert_from_and_to_nodes()

        nodes = (3, 4, 5)
        di_graph.disconnect_nodes(*nodes, fast_mode=fast_mode)

        for depth in range(3):
            complete_paths = complete_graph.all_paths_array(
                from_,
                to,
                depth + 2,
                include_from_and_to=False,
            )
            di_paths = di_graph.all_paths_array(
                from_,
                to,
                depth + 2,
                include_from_and_to=False,
            )
            np.testing.assert_equal(complete_paths, di_paths)

    def test_from_graph(self) -> None:
        graph = CompleteGraph(10)
        assert isinstance(graph, CompleteGraph)
        graph = DiGraph.from_complete_graph(graph)
        assert isinstance(graph, DiGraph)

    def test_all_keyword_only_parameters(self) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(5))

        with pytest.raises(
            TypeError,
            match="takes 3 positional arguments but 4 were given",
        ):
            _ = graph.all_paths(0, 1, 0, True)  # noqa: FBT003

    @pytest.mark.parametrize(
        ("num_nodes", "depth"),
        [
            (15, 2),
            (25, 3),
            (11, 4),
        ],
    )
    def test_all_paths_count_from_complete_graph(
        self,
        num_nodes: int,
        depth: int,
    ) -> None:
        graph = CompleteGraph(num_nodes)
        from_, to = num_nodes, num_nodes + 1
        paths = graph.all_paths(from_, to, depth + 2, include_from_and_to=False)
        num_paths = sum(1 for _ in paths)
        assert num_paths == num_nodes * (num_nodes - 1) ** (depth - 1)
        array = graph.all_paths_array(from_, to, depth + 2, include_from_and_to=False)
        assert array.shape == (num_paths, depth)

    @pytest.mark.parametrize(
        ("num_nodes", "depth"),
        [
            (10, 1),
            (50, 2),
            (10, 3),
        ],
    )
    def test_all_paths_count_from_di_graph(
        self,
        num_nodes: int,
        depth: int,
    ) -> None:
        graph = DiGraph.from_complete_graph(CompleteGraph(num_nodes))
        from_, to = graph.insert_from_and_to_nodes()
        paths = graph.all_paths(from_, to, depth + 2, include_from_and_to=False)
        num_paths = sum(1 for _ in paths)
        assert num_paths == num_nodes * (num_nodes - 1) ** (depth - 1)
        array = graph.all_paths_array(from_, to, depth + 2, include_from_and_to=False)
        assert array.shape == (num_paths, depth)

    @pytest.mark.parametrize(
        ("num_nodes", "depth"),
        [
            (10, 100),
            (50_000, 10),
        ],
    )
    def test_all_paths_count_from_complete_graph_overflow(
        self,
        num_nodes: int,
        depth: int,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        graph = CompleteGraph(num_nodes)
        from_, to = num_nodes, num_nodes + 1

        caplog.clear()

        with caplog.at_level(logging.WARNING):
            _ = graph.all_paths(from_, to, depth + 2, include_from_and_to=False)

            assert (
                "OverflowError: overflow occurred when computing the total number of paths"
                in caplog.text
            )

import numpy as np
from jaxtyping import UInt

class CompleteGraph:
    def __init__(self, num_nodes: int) -> None: ...

class DiGraph:
    @classmethod
    def from_adjacency_matrix(
        cls, adjacency_matrix: UInt[np.ndarray, "num_nodes num_nodes"]
    ) -> DiGraph: ...
    @classmethod
    def from_complete_graph(cls, graph: CompleteGraph) -> DiGraph: ...
    def insert_from_and_to_nodes(self, direct_path: bool = True) -> DiGraph: ...

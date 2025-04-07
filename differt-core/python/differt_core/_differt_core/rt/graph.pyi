# pyright: reportMissingTypeArgument=false
from collections.abc import Iterator, Sized

import numpy as np
from jaxtyping import Bool, UInt

class CompleteGraph:
    @property
    def num_nodes(self) -> int: ...
    def __init__(self, num_nodes: int) -> None: ...
    def all_paths(
        self,
        from_: int,
        to: int,
        depth: int,
        *,
        include_from_and_to: bool = True,
    ) -> AllPathsFromCompleteGraphIter: ...
    def all_paths_array(
        self,
        from_: int,
        to: int,
        depth: int,
        *,
        include_from_and_to: bool = True,
    ) -> UInt[np.ndarray, "num_paths path_depth"]: ...
    def all_paths_array_chunks(
        self,
        from_: int,
        to: int,
        depth: int,
        *,
        include_from_and_to: bool = True,
        chunk_size: int,
    ) -> AllPathsFromCompleteGraphChunksIter: ...

class DiGraph:
    @property
    def num_nodes(self) -> int: ...
    @classmethod
    def from_adjacency_matrix(
        cls,
        adjacency_matrix: Bool[np.ndarray, "num_nodes num_nodes"],
    ) -> DiGraph: ...
    @classmethod
    def from_complete_graph(cls, graph: CompleteGraph) -> DiGraph: ...
    def insert_from_and_to_nodes(
        self,
        *,
        direct_path: bool = True,
        from_adjacency: Bool[np.ndarray, " num_nodes"],
        to_adjacency: Bool[np.ndarray, " num_nodes"],
    ) -> tuple[int, int]: ...
    def disconnect_nodes(self, *nodes: int, fast_mode: bool = True) -> None: ...
    def all_paths(
        self,
        from_: int,
        to: int,
        depth: int,
        *,
        include_from_and_to: bool = True,
    ) -> AllPathsFromDiGraphIter: ...
    def all_paths_array(
        self,
        from_: int,
        to: int,
        depth: int,
        *,
        include_from_and_to: bool = True,
    ) -> UInt[np.ndarray, "num_paths path_depth"]: ...
    def all_paths_array_chunks(
        self,
        from_: int,
        to: int,
        depth: int,
        *,
        include_from_and_to: bool = True,
        chunk_size: int,
    ) -> AllPathsFromDiGraphChunksIter: ...

class AllPathsFromCompleteGraphIter(Iterator, Sized):
    def __iter__(self) -> AllPathsFromCompleteGraphIter: ...
    def __next__(self) -> UInt[np.ndarray, " path_depth"]: ...
    def __len__(self) -> int: ...
    def count(self) -> int: ...

class AllPathsFromCompleteGraphChunksIter(Iterator, Sized):
    def __iter__(self) -> AllPathsFromCompleteGraphChunksIter: ...
    def __next__(self) -> UInt[np.ndarray, "chunk_size path_depth"]: ...
    def __len__(self) -> int: ...
    def count(self) -> int: ...

class AllPathsFromDiGraphIter(Iterator):
    def __iter__(self) -> AllPathsFromDiGraphIter: ...
    def __next__(self) -> UInt[np.ndarray, " path_depth"]: ...
    def count(self) -> int: ...

class AllPathsFromDiGraphChunksIter(Iterator):
    def __iter__(self) -> AllPathsFromDiGraphChunksIter: ...
    def __next__(self) -> UInt[np.ndarray, "chunk_size path_depth"]: ...
    def count(self) -> int: ...

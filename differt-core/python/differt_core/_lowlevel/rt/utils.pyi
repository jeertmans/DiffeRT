from typing import Generic, Protocol, TypeVar

import numpy as np
from jaxtyping import UInt

T = TypeVar("T")

class _SizedIterator(Generic[T], Protocol):
    def __iter__(self) -> _SizedIterator[T]: ...
    def __next__(self) -> T: ...
    def __len__(self) -> int: ...

def generate_all_path_candidates(
    num_primitives: int, order: int
) -> UInt[np.ndarray, "num_path_candidates order"]: ...
def generate_all_path_candidates_iter(
    num_primitives: int, order: int
) -> _SizedIterator[UInt[np.ndarray, " order"]]: ...
def generate_all_path_candidates_chunks_iter(
    num_primitives: int, order: int, chunk_size: int
) -> _SizedIterator[UInt[np.ndarray, "chunk_size order"]]: ...

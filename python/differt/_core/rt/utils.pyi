from collections.abc import Iterator

import numpy as np
from jaxtyping import UInt

def generate_all_path_candidates(
    num_primitives: int, order: int
) -> UInt[np.ndarray, "num_path_candidates order"]: ...
def generate_all_path_candidates_iter(
    num_primitives: int, order: int
) -> Iterator[UInt[np.ndarray, " order"]]: ...

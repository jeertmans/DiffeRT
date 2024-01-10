import numpy as np
from jaxtyping import UInt

def generate_all_path_candidates(
    num_primitives: int, order: int
) -> UInt[np.ndarray, "num_path_candidates order"]: ...

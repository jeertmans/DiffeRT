import pytest

from differt.rt.utils import generate_path_candidates


@pytest.mark.parametrize("num_primitives,order", [(0, 0), (1, 1), (4, 5), (10, 3)])
def test_generate_path_candidates(num_primitives, order, benchmark) -> None:
    _ = benchmark(generate_path_candidates, num_primitives, order)

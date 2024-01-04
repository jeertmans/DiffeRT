import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from differt.rt.utils import generate_all_path_candidates


@pytest.mark.parametrize("num_primitives,order", [(0, 0), (1, 1), (4, 5), (10, 3)])
def test_generate_all_path_candidates(
    num_primitives: int, order: int, benchmark: BenchmarkFixture
) -> None:
    _ = benchmark(generate_all_path_candidates, num_primitives, order)

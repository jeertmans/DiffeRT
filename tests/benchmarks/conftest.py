import pytest_benchmark.plugin as bench_plugin
import pytest_benchmark.stats as bench_stats
from _pytest.config import Config


def pytest_benchmark_generate_json(
    config: Config,
    benchmarks: list[bench_stats.Metadata],
    include_data: bool,
    machine_info: dict[str, str],
    commit_info: dict[str, str],
):
    for bench in benchmarks:
        scale = bench.extra_info.get("scale")
        if bench.stats and scale:
            bench.stats.data = [time / scale for time in bench.stats.data]

    return bench_plugin.pytest_benchmark_generate_json(
        config, benchmarks, include_data, machine_info, commit_info
    )

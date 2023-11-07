"""
The present files contains benchmarks to evaluate performance changes in the tool.

Requirements
------------

First, make sure ``pyperf`` is installed. If you have cloned
the repository, you can use ``poetry install --with test``.

Usage
-----

Running the following command will execute the benchmarks and save the results
for ``bench.json``: ``python benchmark.py -o bench.json``.

To compare multiple benchmark results, you can use
``python -m pyperf compare_to --table bench1.json bench2.json bench3.json ...``.
"""

import pyperf

from differt.rt.utils import generate_path_candidates


def bench_generate_path_candidates(runner: pyperf.Runner) -> None:
    def func():
        generate_path_candidates(num_primitives=5, order=3).block_until_ready()

    runner.bench_func("generate_path_candidates", func)


runner = pyperf.Runner()
bench_generate_path_candidates(runner)

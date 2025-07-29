from typing import Literal

import jax.numpy as jnp
import pytest
from pytest_codspeed import BenchmarkFixture

from differt.rt import (
    fermat_path_on_planar_mirrors,
    image_method,
    triangles_visible_from_vertices,
)
from differt.scene._triangle_scene import TriangleScene

from ..rt.utils import PlanarMirrorsSetup


@pytest.mark.benchmark(group="image_method")
def test_image_method(
    large_random_planar_mirrors_setup: PlanarMirrorsSetup,
    benchmark: BenchmarkFixture,
) -> None:
    setup = large_random_planar_mirrors_setup

    def bench_fun() -> None:
        image_method(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        ).block_until_ready()

    bench_fun()

    _ = benchmark(bench_fun)


@pytest.mark.benchmark(group="fermat_method")
def test_fermat(
    large_random_planar_mirrors_setup: PlanarMirrorsSetup,
    benchmark: BenchmarkFixture,
) -> None:
    setup = large_random_planar_mirrors_setup

    def bench_fun() -> None:
        fermat_path_on_planar_mirrors(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        ).block_until_ready()

    bench_fun()

    _ = benchmark(bench_fun)


@pytest.mark.benchmark(group="triangles_visible_from_vertices")
def test_transmitter_visibility_in_simple_street_canyon_scene(
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
) -> None:
    scene = simple_street_canyon_scene

    def bench_fun() -> None:
        triangles_visible_from_vertices(
            scene.transmitters,
            scene.mesh.triangle_vertices,
        ).block_until_ready()

    bench_fun()

    _ = benchmark(bench_fun)


@pytest.mark.benchmark(group="compute_paths")
@pytest.mark.parametrize("method", ["exhaustive", "hybrid"])
def test_compute_paths_in_simple_street_canyon_scene(
    method: Literal["exhaustive", "hybrid"],
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
) -> None:
    scene = simple_street_canyon_scene.set_assume_quads()

    def bench_fun() -> None:
        num_valid_paths = jnp.array(0, dtype=jnp.int32)
        for order in range(3):
            for paths in scene.compute_paths(
                order=order,
                method=method,
                chunk_size=10_000,
            ):
                num_valid_paths += paths.num_valid_paths
        num_valid_paths.block_until_ready()

    bench_fun()

    _ = benchmark(bench_fun)

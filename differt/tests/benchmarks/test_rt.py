from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray
from pytest_codspeed import BenchmarkFixture

from differt.geometry import fibonacci_lattice
from differt.rt import (
    fermat_path_on_planar_mirrors,
    first_triangles_hit_by_rays,
    image_method,
    rays_intersect_any_triangle,
    triangles_visible_from_vertices,
)
from differt.scene._triangle_scene import TriangleScene

from ..rt.utils import PlanarMirrorsSetup


def random_scene(scene: TriangleScene, *, key: PRNGKeyArray) -> TriangleScene:
    return eqx.tree_at(
        lambda s: s.mesh,
        scene,
        scene.mesh.sample(0.5, by_masking=True, sample_objects=True, key=key),
    )


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


@pytest.mark.benchmark(group="rays_intersect_any_triangle")
def test_rays_intersect_any_triangle(
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(simple_street_canyon_scene, key=key)

    ray_origins = scene.transmitters
    ray_directions = fibonacci_lattice(1_000_000)

    def bench_fun() -> None:
        rays_intersect_any_triangle(
            ray_origins,
            ray_directions,
            scene.mesh.triangle_vertices,
            active_triangles=scene.mesh.mask,
        ).block_until_ready()

    bench_fun()

    _ = benchmark(bench_fun)


@pytest.mark.benchmark(group="triangles_visible_from_vertices")
def test_transmitter_visibility(
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(simple_street_canyon_scene, key=key)

    def bench_fun() -> None:
        triangles_visible_from_vertices(
            scene.transmitters,
            scene.mesh.triangle_vertices,
            active_triangles=scene.mesh.mask,
        ).block_until_ready()

    bench_fun()

    _ = benchmark(bench_fun)


@pytest.mark.benchmark(group="first_triangles_hit_by_rays")
def test_first_triangles_hit_by_rays(
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(simple_street_canyon_scene, key=key)

    ray_origins = scene.transmitters
    ray_directions = fibonacci_lattice(1_000_000)

    def bench_fun() -> None:
        first_triangles_hit_by_rays(
            ray_origins,
            ray_directions,
            scene.mesh.triangle_vertices,
            active_triangles=scene.mesh.mask,
        )[0].block_until_ready()

    bench_fun()

    _ = benchmark(bench_fun)


@pytest.mark.benchmark(group="compute_paths")
@pytest.mark.parametrize("method", ["exhaustive", "hybrid"])
def test_compute_paths(
    method: Literal["exhaustive", "hybrid"],
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(simple_street_canyon_scene.set_assume_quads(), key=key)

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

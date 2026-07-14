from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray
from pytest_codspeed import BenchmarkFixture

from differt.geometry import TracedPaths, fibonacci_lattice
from differt.rt import (
    fermat_path_on_planar_mirrors,
    first_triangle_hit_by_ray,
    image_method,
    ray_intersect_any_triangle,
    triangles_visible_from_vertex,
)
from differt.scene import (
    ExhaustivePathTracer,
    HybridPathTracer,
    TriangleScene,
)

from ..rt.utils import PlanarMirrorsSetup


def random_scene(scene: TriangleScene, *, key: PRNGKeyArray) -> TriangleScene:
    sample_objects = scene.mesh.object_bounds is not None
    return eqx.tree_at(
        lambda s: s.mesh,
        scene,
        scene.mesh.sample(0.5, by_masking=True, sample_objects=sample_objects, key=key),
    )


@pytest.mark.benchmark(group="image_method")
def test_image_method(
    large_random_planar_mirrors_setup: PlanarMirrorsSetup,
    benchmark: BenchmarkFixture,
) -> None:
    setup = large_random_planar_mirrors_setup

    @jax.block_until_ready
    def bench_fun() -> Array:
        return image_method(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        )

    bench_fun()

    benchmark(bench_fun)


@pytest.mark.benchmark(group="fermat_method")
def test_fermat(
    large_random_planar_mirrors_setup: PlanarMirrorsSetup,
    benchmark: BenchmarkFixture,
) -> None:
    setup = large_random_planar_mirrors_setup

    @jax.block_until_ready
    def bench_fun() -> Array:
        return fermat_path_on_planar_mirrors(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        )

    bench_fun()

    benchmark(bench_fun)


@pytest.mark.benchmark(group="ray_intersect_any_triangle")
def test_ray_intersect_any_triangle(
    bench_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(bench_scene, key=key)

    ray_origins = scene.transmitters
    num_rays = 10_000  # TODO: increase this number once the implementation is faster / uses less memory
    ray_directions = fibonacci_lattice(num_rays)

    @jax.block_until_ready
    def bench_fun() -> Array:
        return ray_intersect_any_triangle(
            ray_origins,
            ray_directions,
            scene.mesh.triangle_vertices,
            active_triangles=scene.mesh.mask,
        )

    bench_fun()

    benchmark(bench_fun)


@pytest.mark.benchmark(group="triangles_visible_from_vertex")
def test_transmitter_visibility(
    bench_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(bench_scene, key=key)

    @jax.block_until_ready
    def bench_fun() -> Array:
        return triangles_visible_from_vertex(
            scene.transmitters,
            scene.mesh.triangle_vertices,
            active_triangles=scene.mesh.mask,
            num_rays=10_000,  # TODO: increase this number once the implementation is faster / uses less memory
        )

    bench_fun()

    benchmark(bench_fun)


@pytest.mark.benchmark(group="first_triangle_hit_by_ray")
def test_first_triangle_hit_by_ray(
    bench_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(bench_scene, key=key)

    ray_origins = scene.transmitters
    num_rays = 10_000  # TODO: increase this number once the implementation is faster / uses less memory
    ray_directions = fibonacci_lattice(num_rays)

    @jax.block_until_ready
    def bench_fun() -> tuple[Array, Array]:
        return first_triangle_hit_by_ray(
            ray_origins,
            ray_directions,
            scene.mesh.triangle_vertices,
            active_triangles=scene.mesh.mask,
        )

    bench_fun()

    benchmark(bench_fun)


@pytest.mark.benchmark(group="compute_paths")
@pytest.mark.parametrize(
    "method",
    [pytest.param("exhaustive", id="exhaustive"), pytest.param("hybrid", id="hybrid")],
)
@pytest.mark.parametrize(
    "disconnect_inactive_triangles",
    [pytest.param(False, id="no_disconnect"), pytest.param(True, id="disconnect")],
)
def test_compute_paths(
    method: Literal["exhaustive", "hybrid"],
    disconnect_inactive_triangles: bool,
    bench_scene: TriangleScene,
    benchmark: BenchmarkFixture,
    key: PRNGKeyArray,
) -> None:
    scene = random_scene(bench_scene.set_assume_quads(), key=key)

    if method == "hybrid" and not disconnect_inactive_triangles:
        pytest.skip("Triangles are always disconnected for 'hybrid' method.")

    @jax.block_until_ready
    def bench_fun() -> Array:
        num_valid_paths = jnp.array(0, dtype=jnp.int32)
        for order in [
            0,
            1,
        ]:  # TODO: add higher orders once the implementation is faster / uses less memory
            if method == "hybrid":
                solver = HybridPathTracer(num_rays=10_000)
            else:
                solver = ExhaustivePathTracer(
                    disconnect_inactive_triangles=disconnect_inactive_triangles
                )

            paths = scene.trace_paths(
                order=order,
                solver=solver,
            )
            assert isinstance(paths, TracedPaths)
            num_valid_paths += paths.num_valid_paths
        return num_valid_paths

    bench_fun()

    benchmark(bench_fun)

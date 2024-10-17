import jax
import pytest
from pytest_codspeed import BenchmarkFixture

from differt.rt.fermat import fermat_path_on_planar_mirrors
from differt.rt.image_method import (
    image_method,
)
from differt.rt.utils import triangles_visible_from_vertices
from differt.scene.triangle_scene import TriangleScene

from ..rt.utils import PlanarMirrorsSetup


@pytest.mark.benchmark(group="image_method")
def test_image_method(
    large_random_planar_mirrors_setup: PlanarMirrorsSetup,
    benchmark: BenchmarkFixture,
) -> None:
    setup = large_random_planar_mirrors_setup
    _ = benchmark(
        lambda: image_method(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        ).block_until_ready()
    )


@pytest.mark.benchmark(group="fermat_method")
def test_fermat(
    large_random_planar_mirrors_setup: PlanarMirrorsSetup,
    benchmark: BenchmarkFixture,
) -> None:
    setup = large_random_planar_mirrors_setup
    _ = benchmark(
        lambda: fermat_path_on_planar_mirrors(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        ).block_until_ready()
    )


@pytest.mark.benchmark(group="triangles_visible_from_vertices")
def test_transmitter_visibility_in_simple_street_canyon_scene(
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
) -> None:
    scene = simple_street_canyon_scene
    _ = benchmark(
        lambda: triangles_visible_from_vertices(
            scene.transmitters,
            scene.mesh.triangle_vertices,
        ).block_until_ready()
    )


@pytest.mark.benchmark(group="compute_paths")
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("chunk_size", [None, 20_000])
@pytest.mark.parametrize("assume_quads", [False, True])
def test_compute_paths_in_simple_street_canyon_scene(
    order: int,
    chunk_size: int | None,
    assume_quads: bool,
    simple_street_canyon_scene: TriangleScene,
    benchmark: BenchmarkFixture,
) -> None:
    scene = simple_street_canyon_scene.set_assume_quads(assume_quads)
    if chunk_size:

        @jax.debug_nans(False)  # noqa: FBT003
        def bench_fun() -> None:
            for path in scene.compute_paths(
                order,
                chunk_size=chunk_size,
            ):
                path.vertices.block_until_ready()

    else:

        @jax.debug_nans(False)  # noqa: FBT003
        def bench_fun() -> None:
            scene.compute_paths(
                order,
                chunk_size=None,
            ).vertices.block_until_ready()

    _ = benchmark(bench_fun)

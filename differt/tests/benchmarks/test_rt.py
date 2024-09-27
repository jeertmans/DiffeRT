import pytest
from jaxtyping import PRNGKeyArray
from pytest_benchmark.fixture import BenchmarkFixture

from differt.rt.fermat import fermat_path_on_planar_mirrors
from differt.rt.image_method import (
    image_method,
)

from ..rt.utils import PlanarMirrorsSetup

batches = pytest.mark.parametrize(
    "batch",
    [
        (),
        (10,),
        (
            10,
            20,
            30,
        ),
    ],
)


@pytest.mark.benchmark(group="image_method")
@batches
def test_image_method(
    batch: tuple[int, ...],
    basic_planar_mirrors_setup: PlanarMirrorsSetup,
    key: PRNGKeyArray,
    benchmark: BenchmarkFixture,
) -> None:
    setup = basic_planar_mirrors_setup.broadcast_to(*batch).add_noeffect_noise(key=key)
    _ = benchmark(
        lambda: image_method(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        ).block_until_ready()
    )


@pytest.mark.benchmark(group="fermat_method")
@batches
def test_fermat(
    batch: tuple[int, ...],
    basic_planar_mirrors_setup: PlanarMirrorsSetup,
    key: PRNGKeyArray,
    benchmark: BenchmarkFixture,
) -> None:
    setup = basic_planar_mirrors_setup.broadcast_to(*batch).add_noeffect_noise(key=key)
    _ = benchmark(
        lambda: fermat_path_on_planar_mirrors(
            setup.from_vertices,
            setup.to_vertices,
            setup.mirror_vertices,
            setup.mirror_normals,
        ).block_until_ready()
    )

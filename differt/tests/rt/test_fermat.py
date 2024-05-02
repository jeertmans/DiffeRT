import chex
import pytest

from differt.rt.fermat import fermat_path_on_planar_mirrors

from .utils import PlanarMirrorsSetup


@pytest.mark.parametrize(
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
def test_image_method(batch: tuple[int, ...]) -> None:
    setup = PlanarMirrorsSetup(*batch)
    got = fermat_path_on_planar_mirrors(
        setup.from_vertices,
        setup.to_vertices,
        setup.mirror_vertices,
        setup.mirror_normals,
        steps=10000,
    )
    chex.assert_trees_all_close(got, setup.paths, atol=1e-6)

import chex
import pytest
from jaxtyping import PRNGKeyArray

from differt.rt._fermat import fermat_path_on_planar_mirrors

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
def test_fermat_path_on_planar_mirrors(
    batch: tuple[int, ...],
    basic_planar_mirrors_setup: PlanarMirrorsSetup,
    key: PRNGKeyArray,
) -> None:
    setup = basic_planar_mirrors_setup.broadcast_to(*batch).add_noeffect_noise(key=key)
    got = fermat_path_on_planar_mirrors(
        setup.from_vertices,
        setup.to_vertices,
        setup.mirror_vertices,
        setup.mirror_normals,
        steps=10000,
    )
    chex.assert_trees_all_close(got, setup.paths, atol=1e-4)

import jax.numpy as jnp
import pytest

from .utils import PlanarMirrorsSetup


@pytest.fixture(scope="session")
def basic_planar_mirrors_setup() -> PlanarMirrorsSetup:
    """
    Test setup that looks something like:

                1           3
             ───────     ───────
           0                           5
    (from) x                           x (to)

                   ───────      ───────
                      2            4

    where xs are starting and ending vertices, and '───────' are mirrors.
    """
    return PlanarMirrorsSetup(
        from_vertices=jnp.array([0.0, 0.0, 0.0]),
        to_vertices=jnp.array([1.0, 0.0, 0.0]),
        mirror_vertices=jnp.array(
            [[0.0, +1.0, 0.0], [0.0, -1.0, 0.0], [0.0, +1.0, 0.0], [0.0, -1.0, 0.0]]
        ),
        mirror_normals=jnp.array(
            [[0.0, -1.0, 0.0], [0.0, +1.0, 0.0], [0.0, -1.0, 0.0], [0.0, +1.0, 0.0]],
        ),
        paths=jnp.array(
            [[1.0 / 8.0, +1.0, 0.0], [3.0 / 8.0, -1.0, 0.0], [5.0 / 8.0, +1.0, 0.0], [7.0 / 8.0, -1.0, 0.0]],
        ),
    )

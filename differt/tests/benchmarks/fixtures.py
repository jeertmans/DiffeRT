import jax
import pytest
from jaxtyping import PRNGKeyArray

from ..rt.utils import PlanarMirrorsSetup


@pytest.fixture
def large_random_planar_mirrors_setup(key: PRNGKeyArray) -> PlanarMirrorsSetup:
    num_mirrors = 3
    num_path_candidates = 10_000

    key_from, key_to, key_m_vertices, key_m_normals, key_paths = jax.random.split(
        key, 5
    )

    from_vertices = jax.random.uniform(key_from, (3,))
    to_vertices = jax.random.uniform(key_to, (3,))
    mirror_vertices = jax.random.uniform(
        key_m_vertices, (num_path_candidates, num_mirrors, 3)
    )
    mirror_normals = jax.random.uniform(
        key_m_normals, (num_path_candidates, num_mirrors, 3)
    )
    paths = jax.random.uniform(key_paths, (num_path_candidates, num_mirrors, 3))

    return PlanarMirrorsSetup(
        from_vertices, to_vertices, mirror_vertices, mirror_normals, paths
    )

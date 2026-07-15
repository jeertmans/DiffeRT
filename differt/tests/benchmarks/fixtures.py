from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry import Mesh, normalize
from differt.scene import (
    Scene,
    get_sionna_scene,
)

from ..rt.utils import PlanarMirrorsSetup


@pytest.fixture
def large_random_planar_mirrors_setup(key: PRNGKeyArray) -> PlanarMirrorsSetup:
    num_mirrors = 8
    num_path_candidates = 10_000

    key_from, key_to, key_m_vertices, key_m_normals, key_paths = jax.random.split(
        key, 5
    )

    from_vertices = jax.random.uniform(key_from, (3,))
    to_vertices = jax.random.uniform(key_to, (3,))
    mirror_vertices = jax.random.uniform(
        key_m_vertices, (num_path_candidates, num_mirrors, 3)
    )
    mirror_normals = normalize(
        jax.random.normal(key_m_normals, (num_path_candidates, num_mirrors, 3))
    )[0]
    paths = jax.random.uniform(key_paths, (num_path_candidates, num_mirrors, 3))

    return PlanarMirrorsSetup(
        from_vertices, to_vertices, mirror_vertices, mirror_normals, paths
    )


@pytest.fixture(params=["small", "medium"])
def bench_scene(request: pytest.FixtureRequest, sionna_folder: Path) -> Scene:
    if request.param == "small":
        file = get_sionna_scene("simple_street_canyon", folder=sionna_folder)
        scene = Scene.load_xml(file)
        scene = eqx.tree_at(
            lambda s: s.transmitters, scene, jnp.array([-22.0, 0.0, 32.0])
        )
        return eqx.tree_at(lambda s: s.receivers, scene, jnp.array([+22.0, 0.0, 32.0]))
    if request.param == "medium":
        # Root of the repo is 3 parents up from differt/tests/benchmarks/fixtures.py
        root = Path(__file__).parents[3]
        file = root / "docs" / "source" / "notebooks" / "bruxelles.obj"
        scene = Scene(mesh=Mesh.load_obj(str(file)))
        # Add a transmitter at some reasonable position
        vertices = scene.mesh.vertices
        center = jnp.mean(vertices, axis=0)
        scene = eqx.tree_at(
            lambda s: s.transmitters, scene, center[None, :] + jnp.array([0, 0, 10.0])
        )
        return eqx.tree_at(
            lambda s: s.receivers, scene, center[None, :] + jnp.array([10.0, 0, 1.5])
        )

    msg = f"Unknown scene variant: {request.param}"
    raise ValueError(msg)

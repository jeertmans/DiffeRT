import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import PRNGKeyArray

from differt.geometry import assemble_paths, normalize
from differt.rt._fermat import (
    fermat_path_on_linear_objects,
    fermat_path_on_planar_mirrors,
)
from differt.utils import dot

from .utils import PlanarMirrorsSetup


def test_fermat_path_on_linear_objects(
    key: PRNGKeyArray,
) -> None:
    from_vertex = jnp.array([-2.0, 0.0, 0.0])
    to_vertex = jnp.array([0.0, 0.0, 0.0])
    edge_origin = jnp.array([-1.0, -0.5, 0.5])
    edge_vectors = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],  # Adding zeros for higher dimension
    ])
    mirror_origin = jnp.array(
        [1.0, 0.0, 0.0],
    )
    mirror_vectors = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    object_vectors = jnp.stack((edge_vectors, mirror_vectors), axis=0)
    object_origins = jnp.stack((edge_origin, mirror_origin), axis=0)
    # Adding noise to the object origins (mimicking a different initialization)
    object_origins += (jax.random.uniform(key, shape=(2, 1)) * object_vectors).sum(
        axis=-2
    )
    got = fermat_path_on_linear_objects(
        from_vertex,
        to_vertex,
        object_origins,
        object_vectors,
        steps=10000,
        optimizer=optax.lbfgs(),
    )
    expected = jnp.array([[-1.0, 0.0, 0.5], [1.0, 0.0, 0.5 / 3]])
    chex.assert_trees_all_close(got, expected, atol=1e-5)


@pytest.mark.parametrize("num_dims", [1, 2, 3])
def test_fermat_path_diffraction_keller_cone(num_dims: int) -> None:
    h = jnp.linspace(-0.5, 0.5, 5)
    edge_origins = jnp.array([[-0.5, 0.5, -0.5]])
    edge_vectors = jnp.concatenate(
        (
            jnp.array([[[0.0, 0.0, 1.0]]]),
            jnp.zeros((
                1,
                num_dims - 1,
                3,
            )),  # Adding zeros for higher dimensions does not affect the test
        ),
        axis=-2,
    )
    transmitters = jnp.array([-1.0, -0.5, 0.0])
    receivers = jnp.stack((jnp.zeros_like(h), jnp.ones_like(h), h), axis=-1)
    paths = fermat_path_on_linear_objects(
        transmitters, receivers, edge_origins, edge_vectors
    )

    paths = assemble_paths(
        transmitters.reshape(1, 1, 3),
        paths,
        receivers.reshape(-1, 1, 3),
    )
    rays = normalize(jnp.diff(paths, axis=-2))[0]
    i = rays[..., 0, :]
    d = rays[..., 1, :]
    chex.assert_trees_all_close(
        dot(i, edge_vectors[:, 0, :]), dot(d, edge_vectors[:, 0, :])
    )


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
    chex.assert_trees_all_close(got, setup.paths, atol=1e-3)

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest
from chex import Array

from differt.rt.image_method import image_of_vertices_with_respect_to_mirrors
from tests.utils import random_inputs


def test_image_of_vertices_with_respect_to_mirrors() -> None:
    vertices = jnp.array([[+0.0, +0.0, +1.0], [+1.0, +2.0, +3.0]])
    expected = jnp.array([[+0.0, +0.0, -1.0], [+1.0, +2.0, -3.0]])
    mirror_vertex = jnp.array([0.0, 0.0, 0.0])
    mirror_normal = jnp.array([0.0, 0.0, 1.0])
    n = vertices.shape[0]
    mirror_vertices = jnp.tile(mirror_vertex, (n, 1))
    mirror_normals = jnp.tile(mirror_normal, (n, 1))
    got = image_of_vertices_with_respect_to_mirrors(
        vertices, mirror_vertices, mirror_normals
    )
    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    ("vertices,mirror_vertices,mirror_normals,expectation"),
    [
        ((10, 3), (10, 3), (10, 3), does_not_raise()),
        (
            (10, 3),
            (20, 3),
            (10, 3),
            pytest.raises(TypeError),
        ),
        (
            (10, 3),
            (10, 4),
            (10, 3),
            pytest.raises(TypeError),
        ),
        ((3,), (3,), (3,), does_not_raise()),
    ],
)
@random_inputs("vertices", "mirror_vertices", "mirror_normals")
def test_image_of_vertices_with_respect_to_mirrors_random_inputs(
    vertices: Array,
    mirror_vertices: Array,
    mirror_normals: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = image_of_vertices_with_respect_to_mirrors(
            vertices, mirror_vertices, mirror_normals
        )
        if len(vertices.shape) == 2:
            for i in range(vertices.shape[0]):
                incident = vertices[i, :] - mirror_vertices[i, :]
                expected = (
                    vertices[i, :]
                    - 2.0
                    * jnp.dot(incident, mirror_normals[i, :])
                    * mirror_normals[i, :]
                )
                chex.assert_trees_all_close(got[i, :], expected, rtol=1e-5)
        else:
            incident = vertices - mirror_vertices
            expected = (
                vertices - 2.0 * jnp.dot(incident, mirror_normals) * mirror_normals
            )
            chex.assert_trees_all_close(got, expected, rtol=1e-5)


# def test_image_of_vertices_with_respect_to_mirrors() -> None:
#    vertices = jnp.array([[+0.0, +0.0, +1.0], [+1.0, +2.0, +3.0]])
#    expected = jnp.array([[+0.0, +0.0, -1.0], [+1.0, +2.0, -3.0]])
#    mirror_vertex = jnp.array([0.0, 0.0, 0.0])
#    mirror_normal = jnp.array([0.0, 0.0, 1.0])
#    n = vertices.shape[0]
#    mirror_vertices = jnp.tile(mirror_vertex, (n, 1))
#    mirror_normals = jnp.tile(mirror_normal, (n, 1))
#    got = image_of_vertices_with_respect_to_mirrors(
#        vertices, mirror_vertices, mirror_normals
#    )
#    chex.assert_trees_all_close(got, expected)

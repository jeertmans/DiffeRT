from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from differt.rt.image_method import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    image_method,
    image_of_vertices_with_respect_to_mirrors,
    intersection_of_line_segments_with_planes,
)

from ..utils import random_inputs
from .utils import PlanarMirrorsSetup


def test_image_of_vertices_with_respect_to_mirrors() -> None:
    vertices = jnp.array([[+0.0, +0.0, +1.0], [+1.0, +2.0, +3.0]])
    expected = jnp.array([[+0.0, +0.0, -1.0], [+1.0, +2.0, -3.0]])
    mirror_vertex = jnp.array([0.0, 0.0, 0.0])
    mirror_normal = jnp.array([0.0, 0.0, 1.0])
    n = vertices.shape[0]
    mirror_vertices = jnp.tile(mirror_vertex, (n, 1))
    mirror_normals = jnp.tile(mirror_normal, (n, 1))
    got = image_of_vertices_with_respect_to_mirrors(
        vertices,
        mirror_vertices,
        mirror_normals,
    )
    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    ("vertices,mirror_vertices,mirror_normals,expectation"),
    [
        ((20, 10, 3), (20, 10, 3), (20, 10, 3), does_not_raise()),
        ((10, 3), (10, 3), (10, 3), does_not_raise()),
        ((3,), (3,), (3,), does_not_raise()),
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
            vertices,
            mirror_vertices,
            mirror_normals,
        )
        for i in np.ndindex(vertices.shape[:-1]):
            index = (*i, slice(0, None))  # [i, :]
            incident = vertices[index] - mirror_vertices[index]
            expected = (
                vertices[index]
                - 2.0 * jnp.dot(incident, mirror_normals[index]) * mirror_normals[index]
            )
            chex.assert_trees_all_close(got[index], expected, rtol=1e-5)


def test_intersection_of_line_segments_with_planes() -> None:
    segment_starts = jnp.array(
        [[-1.0, +1.0, +0.0], [-2.0, +1.0, +0.0], [-3.0, +1.0, +0.0]],
    )
    expected = jnp.array([[+0.5, +0.0, +0.0], [+0.0, +0.0, +0.0], [-0.5, +0.0, +0.0]])
    segment_end = jnp.array([2.0, -1.0, 0.0])
    plane_vertex = jnp.array([0.0, 0.0, 0.0])
    plane_normal = jnp.array([0.0, 1.0, 0.0])

    n = segment_starts.shape[0]
    segment_ends = jnp.tile(segment_end, (n, 1))
    plane_vertices = jnp.tile(plane_vertex, (n, 1))
    plane_normals = jnp.tile(plane_normal, (n, 1))
    got = intersection_of_line_segments_with_planes(
        segment_starts,
        segment_ends,
        plane_vertices,
        plane_normals,
    )
    chex.assert_trees_all_close(got, expected)


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
    got = image_method(
        setup.from_vertices,
        setup.to_vertices,
        setup.mirror_vertices,
        setup.mirror_normals,
    )
    chex.assert_trees_all_close(got, setup.paths)


@pytest.mark.parametrize(
    ("vertices,mirror_vertices,mirror_normals,expectation"),
    [
        ((12, 3), (10, 3), (10, 3), does_not_raise()),
        ((4, 12, 3), (4, 10, 3), (4, 10, 3), does_not_raise()),
        ((6, 7, 12, 3), (6, 7, 10, 3), (6, 7, 10, 3), does_not_raise()),
        (
            (12, 3),
            (10, 3),
            (11, 3),
            pytest.raises(TypeError),
        ),
        (
            (10, 3),
            (12, 4),
            (12, 3),
            pytest.raises(TypeError),
        ),
    ],
)
@random_inputs("vertices", "mirror_vertices", "mirror_normals")
def test_consecutive_vertices_are_on_same_side_of_mirrors(
    vertices: Array,
    mirror_vertices: Array,
    mirror_normals: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        got = consecutive_vertices_are_on_same_side_of_mirrors(
            vertices,
            mirror_vertices,
            mirror_normals,
        )
        chex.assert_trees_all_equal_shapes(got, mirror_vertices[..., 0])

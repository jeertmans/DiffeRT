from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray

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
    expected = jnp.array([[+0.0, +0.0, -1.0], [+1.0, +2.0, -3.0]]).reshape(2, 1, 3)
    mirror_vertices = jnp.array([[0.0, 0.0, 0.0]])
    mirror_normals = jnp.array([[0.0, 0.0, 1.0]])
    got = image_of_vertices_with_respect_to_mirrors(
        vertices,
        mirror_vertices,
        mirror_normals,
    )
    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    ("vertices", "mirror_vertices", "mirror_normals", "expectation"),
    [
        ((20, 3), (10, 3), (10, 3), does_not_raise()),
        ((10, 3), (1, 3), (1, 3), does_not_raise()),
        ((3,), (1, 3), (1, 3), does_not_raise()),
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
        ).reshape(-1, *mirror_vertices.shape)
        for i, vertex in enumerate(vertices.reshape(-1, 3)):
            incident = vertex[None, ...] - mirror_vertices
            expected = (
                vertex[None, ...]
                - 2.0
                * jnp.sum(incident * mirror_normals, axis=-1, keepdims=True)
                * mirror_normals
            )
            chex.assert_trees_all_close(got[i, :, :], expected, rtol=1e-5)


def test_intersection_of_line_segments_with_planes() -> None:
    segment_starts = jnp.array(
        [[-1.0, +1.0, +0.0], [-2.0, +1.0, +0.0], [-3.0, +1.0, +0.0]],
    )
    expected = jnp.array([
        [+0.5, +0.0, +0.0],
        [+0.0, +0.0, +0.0],
        [-0.5, +0.0, +0.0],
    ]).reshape(3, 1, 3)
    segment_ends = jnp.broadcast_to(jnp.array([[2.0, -1.0, 0.0]]), segment_starts.shape)
    plane_vertices = jnp.array([[0.0, 0.0, 0.0]])
    plane_normals = jnp.array([[0.0, 1.0, 0.0]])
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
@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_image_method(
    batch: tuple[int, ...],
    scale: float,
    basic_planar_mirrors_setup: PlanarMirrorsSetup,
    key: PRNGKeyArray,
) -> None:
    setup = basic_planar_mirrors_setup.broadcast_to(*batch).add_noeffect_noise(
        scale=scale, key=key
    )
    got = image_method(
        setup.from_vertices,
        setup.to_vertices,
        setup.mirror_vertices,
        setup.mirror_normals,
    )
    chex.assert_trees_all_close(got, setup.paths)


@pytest.mark.parametrize(
    ("vertices", "mirror_vertices", "mirror_normals", "expectation"),
    [
        ((12, 3), (10, 3), (10, 3), does_not_raise()),
        ((4, 12, 3), (10, 3), (10, 3), does_not_raise()),
        ((6, 7, 12, 3), (10, 3), (10, 3), does_not_raise()),
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
        chex.assert_axis_dimension(got, -1, mirror_vertices.shape[0])
        chex.assert_trees_all_equal_shapes(got[..., 0], vertices[..., 0, 0])

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray

from differt.geometry import normalize
from differt.rt._image_method import (
    consecutive_vertices_are_on_same_side_of_mirrors,
    image_method,
    image_of_vertices_with_respect_to_mirrors,
    intersection_of_rays_with_planes,
)

from ..utils import random_inputs
from .utils import PlanarMirrorsSetup


def test_image_of_vertices_with_respect_to_mirrors() -> None:
    vertices = jnp.array([[+0.0, +0.0, +1.0], [+1.0, +2.0, +3.0]])
    expected = jnp.array([[+0.0, +0.0, -1.0], [+1.0, +2.0, -3.0]])
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
        ((10, 3), (1, 3), (1, 3), does_not_raise()),
        ((10, 3), (10, 1, 3), (10, 1, 3), does_not_raise()),
        ((10, 3), (10, 1, 3), (1, 1, 3), does_not_raise()),
        ((1, 3), (10, 1, 3), (1, 1, 3), does_not_raise()),
        ((3,), (1, 3), (1, 3), does_not_raise()),
        ((20, 3), (10, 3), (10, 3), pytest.raises(TypeError)),
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
        ).reshape(-1, 3)
        vertices, mirror_vertices, mirror_normals = jnp.broadcast_arrays(
            vertices, mirror_vertices, mirror_normals
        )
        for i, (vertex, mirror_vertex, mirror_normal) in enumerate(
            zip(
                vertices.reshape(-1, 3),
                mirror_vertices.reshape(-1, 3),
                mirror_normals.reshape(-1, 3),
                strict=False,
            )
        ):
            incident = vertex - mirror_vertex
            expected = vertex - 2.0 * jnp.sum(incident * mirror_normal) * mirror_normal
            chex.assert_trees_all_close(got[i, :], expected, rtol=1e-5)


def test_intersection_of_rays_with_planes() -> None:
    ray_origins = jnp.array(
        [[-1.0, +1.0, +0.0], [-2.0, +1.0, +0.0], [-3.0, +1.0, +0.0]],
    )
    expected = jnp.array([
        [+0.5, +0.0, +0.0],
        [+0.0, +0.0, +0.0],
        [-0.5, +0.0, +0.0],
    ])
    ray_ends = jnp.broadcast_to(jnp.array([[2.0, -1.0, 0.0]]), ray_origins.shape)
    ray_directions = ray_ends - ray_origins
    plane_vertices = jnp.array([[0.0, 0.0, 0.0]])
    plane_normals = jnp.array([[0.0, 1.0, 0.0]])
    got = intersection_of_rays_with_planes(
        ray_origins,
        ray_directions,
        plane_vertices,
        plane_normals,
    )
    chex.assert_trees_all_close(got, expected)


def test_intersection_of_rays_with_planes_parallel() -> None:
    ray_origins = jnp.array(
        [[-1.0, +1.0, +0.0], [-2.0, +1.0, +0.0], [-3.0, +1.0, +0.0]],
    )
    ray_ends = jnp.broadcast_to(jnp.array([[2.0, -1.0, 0.0]]), ray_origins.shape)
    ray_directions = ray_ends - ray_origins
    plane_vertices = jnp.array([[0.0, 0.0, -1.0]])
    plane_normals = jnp.array([[0.0, 0.0, 1.0]])
    got = intersection_of_rays_with_planes(
        ray_origins,
        ray_directions,
        plane_vertices,
        plane_normals,
    )
    expected = jnp.full_like(got, jnp.inf)
    chex.assert_trees_all_close(got, expected)

    # Check that we have non-NaNs gradient
    grads = jax.grad(lambda *args: intersection_of_rays_with_planes(*args).sum())(
        ray_origins,
        ray_directions,
        plane_vertices,
        plane_normals,
    )

    assert not jnp.isnan(grads).any()

    # Ray origins are on the plane

    plane_vertices = jnp.array([[0.0, 0.0, 0.0]])
    got = intersection_of_rays_with_planes(
        ray_origins,
        ray_directions,
        plane_vertices,
        plane_normals,
    )
    expected = ray_origins
    chex.assert_trees_all_close(got, expected)


@pytest.mark.parametrize(
    (
        "ray_origins",
        "ray_directions",
        "plane_vertices",
        "plane_normals",
        "expectation",
    ),
    [
        ((10, 3), (10, 3), (1, 3), (1, 3), does_not_raise()),
        ((3,), (3,), (3,), (3,), does_not_raise()),
        ((10, 3), (1, 10, 3), (1, 1, 3), (10, 1, 3), does_not_raise()),
        ((10, 3), (1, 10, 3), (1, 1, 3), (10, 2, 3), pytest.raises(TypeError)),
        ((20, 3), (10, 3), (10, 3), (10, 3), pytest.raises(TypeError)),
    ],
)
@random_inputs("ray_origins", "ray_directions", "plane_vertices", "plane_normals")
def test_intersection_of_rays_with_planes_random_inputs(
    ray_origins: Array,
    ray_directions: Array,
    plane_vertices: Array,
    plane_normals: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        _ = intersection_of_rays_with_planes(
            ray_origins,
            ray_directions,
            plane_vertices,
            plane_normals,
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
    ("from_vertices", "to_vertices", "mirror_vertices", "mirror_normals"),
    [
        ((3), (3,), (1, 3), (1, 3)),
        ((3), (3,), (10, 3), (10, 3)),
        ((5, 3), (3,), (10, 3), (10, 3)),
    ],
)
@random_inputs("from_vertices", "to_vertices", "mirror_vertices", "mirror_normals")
def test_image_method_return_vertices_on_mirrors(
    from_vertices: Array,
    to_vertices: Array,
    mirror_vertices: Array,
    mirror_normals: Array,
) -> None:
    mirror_normals = normalize(mirror_normals)[0]
    paths = image_method(
        from_vertices,
        to_vertices,
        mirror_vertices,
        mirror_normals,
    )
    vectors = paths - mirror_vertices
    # Dot product should be zero as vectors are perpendicular to normals
    got = jnp.sum(vectors * mirror_normals, axis=-1)
    got = jnp.nan_to_num(got, posinf=0.0, neginf=0.0)  # Remove 'inf' values

    excepted = jnp.zeros_like(got)
    chex.assert_trees_all_close(got, excepted, atol=1e-4)


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
        (
            (12, 3),
            (11, 4),
            (11, 3),
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

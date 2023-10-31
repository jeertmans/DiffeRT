"""
Path tracing utilities that utilize the Image Method.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def image_of_vertices_with_respect_to_mirrors(
    vertices: Float[Array, "*batch 3"],
    mirror_vertices: Float[Array, "*batch 3"],
    mirror_normals: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """
    Return the image of vertices with respect to mirrors.
    """
    incident = vertices - mirror_vertices  # incident vectors
    return (
        vertices
        - 2.0
        * jnp.sum(incident * mirror_normals, axis=-1, keepdims=True)
        * mirror_normals
    )


@jaxtyped
@typechecker
def intersection_with_mirrors(
    vertices: Float[Array, "N 3"],
    mirror_vertices: Float[Array, "N 3"],
    mirror_normals: Float[Array, "N 3"],
) -> Float[Array, "N 3"]:
    """
    Return the image of vertices with respect to mirrors.
    """
    incident = vertices - mirror_vertices  # incident vectors
    return (
        vertices
        - 2.0
        * jnp.sum(incident * mirror_normals, axis=-1, keepdims=True)
        * mirror_normals
    )


@jaxtyped
@typechecker
def image_method(
    from_vertices: Float[Array, "N 3"],
    to_vertices: Float[Array, "N 3"],
    mirror_vertices: Float[Array, "N num_mirrors 3"],
    mirror_normals: Float[Array, "N num_mirrors 3"],
) -> Float[Array, "N num_mirrors 3"]:
    """
    Return the ray path between pair of vertices, that reflect on
    a given list of mirrors in between.
    """

    def forward(vertices, carry):
        mirror_vertices, mirror_normals = carry
        images = image_of_vertices_with_respect_to_mirrors(
            vertices, mirror_vertices, mirror_normals
        )
        return images, images

    def backward(vertices, carry):
        mirror_vertices, mirror_normals, images = carry
        project_
        p = wall.origin()
        n = wall.normal()
        u = point - image
        v = p - point
        un = jnp.dot(u, n)
        vn = jnp.dot(v, n)
        # Avoid division by zero
        inc = jnp.where(un == 0.0, 0.0, vn * u / un)
        point = point + inc
        return point, point

    _, images = jax.lax.scan(
        forward, init=from_vertices, xs=(mirror_vertices, mirror_normals)
    )
    _, points = jax.lax.scan(
        backward, init=to_vertices, xs=(mirror_vertices, mirror_normals, images)
    )

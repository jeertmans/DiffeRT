"""
Path tracing utilities that utilize the Image Method.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def image_of_vertices_with_respect_to_mirrors(
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


def image_method():
    pass

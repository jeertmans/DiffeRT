"""
Path tracing utilities that utilize the Image Method.

The Image Method is a very simple but effective path tracing technique
that can rapidly compute a ray path undergoing a series
of specular reflections on a pre-defined list of mirrors.

The method assumes infinitely long mirrors, and will return invalid
paths in some degenerated cases such as consecutive colinear mirrors,
or impossible configurations. It is the user's responsibility to make
sure that the returned path is correct.

Otherwise, the returned path will, for each reflection, have equal angles
of incidence and of reflection.

Examples:
    The following image shows how the Image Method (IM) can be applied
    to find a path between two nodes (i.e., BS and UE).

    .. figure:: ../_static/image-method.svg
        :width: 70%
        :align: center
        :alt: Image Method example.

        Example application of IM in RT. The method determines the only
        valid path that can be taken to join BS and UE with, in between, reflection with
        two mirrors (the interaction order is important). First, the consecutive images
        of the BS are determined through each mirror, using line symmetry. Second,
        intersections with mirrors are computed backward, `i.e.`, from last mirror to
        first, by joining the UE, then the intersections points, with the images of the
        BS. Finally, the valid path can be obtained by joining BS, the intermediary
        intersection points, and the UE :cite:`mpt-eucap2023{fig. 5}`.
"""

import chex
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=typechecker)
def image_of_vertices_with_respect_to_mirrors(
    vertices: Float[Array, "*batch 3"],
    mirror_vertices: Float[Array, "*batch 3"],
    mirror_normals: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """
    Return the image of vertices with respect to mirrors.

    Args:
        vertices: An array of vertices that will be mirrored.
        mirror_vertices: An array of mirror vertices. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normals: An array of mirror normals, where each normal has a unit
            length and if perpendicular to the corresponding mirror.

    Return:
        An array of image vertices.

    Examples:
        In the following example, we show how to compute the images of
        a batch of random vertices. Here, normal vectors do not have a unit length,
        but they should have if you want an interpretable result.

        >>> from differt.rt.image_method import (
        ...     image_of_vertices_with_respect_to_mirrors,
        ... )
        >>>
        >>> key = jax.random.PRNGKey(0)
        >>> (
        ...     key0,
        ...     key1,
        ...     key2,
        ... ) = jax.random.split(key, 3)
        >>> batch = (10, 20, 30)
        >>> vertices = jax.random.uniform(
        ...     key0,
        ...     (*batch, 3),
        ... )
        >>> mirror_vertices = jax.random.uniform(
        ...     key1,
        ...     (*batch, 3),
        ... )
        >>> mirror_normals = jax.random.uniform(
        ...     key2,
        ...     (*batch, 3),
        ... )
        >>> images = image_of_vertices_with_respect_to_mirrors(
        ...     vertices,
        ...     mirror_vertices,
        ...     mirror_normals,
        ... )
        >>> images.shape
        (10, 20, 30, 3)

    """
    incident = vertices - mirror_vertices  # incident vectors
    return (
        vertices
        - 2.0
        * jnp.sum(incident * mirror_normals, axis=-1, keepdims=True)
        * mirror_normals
    )


@jax.jit
@jaxtyped(typechecker=typechecker)
def intersection_of_line_segments_with_planes(
    segment_starts: Float[Array, "*batch 3"],
    segment_ends: Float[Array, "*batch 3"],
    plane_vertices: Float[Array, "*batch 3"],
    plane_normals: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """
    Return the intersection points between line segments and (infinite) planes.

    If a line segment is parallel to the corresponding plane, then
    the corresponding vertex in ``from_vertices`` will be returned.

    Args:
        segment_starts: An array of vertices describing the start of line
            segments.

            .. note::

                ``segment_starts`` and ``segment_ends`` are interchangeable.
        segment_ends: An array of vertices describing the end of line segments.
        plane_vertices: An array of plane vertices. For each plane, any
            vertex on this plane can be used.
        plane_normals: an array of plane normals, where each normal has a unit
            length and if perpendicular to the corresponding plane.

    Return:
        An array of intersection vertices.
    """
    u = segment_ends - segment_starts
    v = plane_vertices - segment_starts
    un = jnp.sum(u * plane_normals, axis=-1, keepdims=True)
    vn = jnp.sum(v * plane_normals, axis=-1, keepdims=True)
    offset = jnp.where(un == 0.0, 0.0, vn * u / un)
    return segment_starts + offset


@jax.jit
@jaxtyped(typechecker=typechecker)
def image_method(
    from_vertices: Float[Array, "*batch 3"],
    to_vertices: Float[Array, "*batch 3"],
    mirror_vertices: Float[Array, "*batch num_mirrors 3"],
    mirror_normals: Float[Array, "*batch num_mirrors 3"],
) -> Float[Array, "*batch num_mirrors 3"]:
    """
    Return the ray paths between pairs of vertices, that reflect on a given list of mirrors in between.

    Args:
        from_vertices: An array of ``from`` vertices, i.e., vertices from which the
            ray paths start. In a radio communications context, this is usually
            an array of transmitters.
        to_vertices: An array of ``to`` vertices, i.e., vertices to which the
            ray paths end. In a radio communications context, this is usually
            an array of receivers.
        mirror_vertices: An array of mirror vertices. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normals: An array of mirror normals, where each normal has a unit
            length and if perpendicular to the corresponding mirror.

    Return:
        An array of ray paths obtained with the image method.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`jax.numpy.concatenate`:

        .. code-block:: python

            paths = image_method(
                from_vertices,
                to_vertices,
                mirror_vertices,
                mirror_normals,
            )

            full_paths = jnp.concatenate(
                (jnp.expand_dims(from_vertices, -2), got, jnp.expand_dims(to_vertices, -2)),
                axis=-2,
            )

    """
    # Put num_mirrors axis as leading axis
    mirror_vertices = jnp.moveaxis(mirror_vertices, -2, 0)
    mirror_normals = jnp.moveaxis(mirror_normals, -2, 0)

    @jaxtyped(typechecker=typechecker)
    def forward(
        carry: Float[Array, "*batch 3"],
        x: tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        """Perform forward pass on vertices by computing consecutive images."""
        vertices = carry
        mirror_vertices, mirror_normals = x
        images = image_of_vertices_with_respect_to_mirrors(
            vertices, mirror_vertices, mirror_normals
        )
        return images, images

    @jaxtyped(typechecker=typechecker)
    def backward(
        carry: Float[Array, "*batch 3"],
        x: tuple[
            Float[Array, "*batch 3"], Float[Array, "*batch 3"], Float[Array, "*batch 3"]
        ],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        """Perform backward pass on images by computing the intersection with mirrors."""
        vertices = carry
        mirror_vertices, mirror_normals, images = x

        intersections = intersection_of_line_segments_with_planes(
            vertices, images, mirror_vertices, mirror_normals
        )
        return intersections, intersections

    _, images = jax.lax.scan(
        forward, init=from_vertices, xs=(mirror_vertices, mirror_normals)
    )
    _, paths = jax.lax.scan(
        backward,
        init=to_vertices,
        xs=(mirror_vertices, mirror_normals, images),
        reverse=True,
    )

    return jnp.moveaxis(paths, 0, -2)


@jax.jit
@jaxtyped(typechecker=typechecker)
def consecutive_vertices_are_on_same_side_of_mirrors(
    vertices: Float[Array, "*batch num_vertices 3"],
    mirror_vertices: Float[Array, "*batch num_mirrors 3"],
    mirror_normals: Float[Array, "*batch num_mirrors 3"],
) -> Bool[Array, "*batch num_mirrors"]:
    """
    Check if consecutive vertices, but skiping one every other vertex, are on the same side of a given mirror. The number of vertices ``num_vertices`` must be equal to ``num_mirrors + 2``.

    This check is needed after using :func:`image_method` because it can return
    vertices that are behind a mirror, which causes the path to go through this
    mirror, and is someone we want to avoid.

    Args:
        vertices: An array of vertices, usually describing ray paths.
        mirror_vertices: An array of mirror vertices. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normals: An array of mirror normals, where each normal has a unit
            length and is perpendicular to the corresponding mirror.

    Return:
        A boolean array indicating whether pairs of consecutive vertices
        are on the same side of the corresponding mirror.
    """
    chex.assert_axis_dimension(vertices, -2, mirror_vertices.shape[-2] + 2)

    v_prev = vertices[..., :-2, :] - mirror_vertices
    v_next = vertices[..., +2:, :] - mirror_vertices

    d_prev = jnp.sum(v_prev * mirror_normals, axis=-1)
    d_next = jnp.sum(v_next * mirror_normals, axis=-1)

    return (d_prev * d_next) >= 0.0

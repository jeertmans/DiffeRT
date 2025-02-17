# ruff: noqa: ERA001

from functools import partial

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from differt.utils import dot


@partial(jax.jit, inline=True)
def image_of_vertices_with_respect_to_mirrors(
    vertices: Float[ArrayLike, "*#batch 3"],
    mirror_vertices: Float[ArrayLike, "*#batch 3"],
    mirror_normals: Float[ArrayLike, "*#batch 3"],
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

    Returns:
        An array of image vertices.

    Examples:
        In the following example, we show how to compute the images of
        a batch of random vertices. Here, normal vectors do not have a unit length,
        but they should have if you want an interpretable result.

        >>> from differt.rt import (
        ...     image_of_vertices_with_respect_to_mirrors,
        ... )
        >>>
        >>> key = jax.random.key(0)
        >>> (
        ...     key0,
        ...     key1,
        ...     key2,
        ... ) = jax.random.split(key, 3)
        >>> *batch, num_mirrors = (10, 20, 30)
        >>> vertices = jax.random.uniform(
        ...     key0,
        ...     (*batch, 1, 3),  # 1 so that it can broadcast with mirrors' shape
        ... )
        >>> mirror_vertices = jax.random.uniform(
        ...     key1,
        ...     (num_mirrors, 3),
        ... )
        >>> mirror_normals = jax.random.uniform(
        ...     key2,
        ...     (num_mirrors, 3),
        ... )
        >>> images = image_of_vertices_with_respect_to_mirrors(
        ...     vertices,
        ...     mirror_vertices,
        ...     mirror_normals,
        ... )
        >>> images.shape
        (10, 20, 30, 3)

    """
    vertices = jnp.asarray(vertices)
    mirror_vertices = jnp.asarray(mirror_vertices)
    mirror_normals = jnp.asarray(mirror_normals)

    # [*batch num_mirrors ]
    incident = vertices - mirror_vertices  # incident vectors
    return (
        vertices - 2.0 * dot(incident, mirror_normals, keepdims=True) * mirror_normals
    )


@partial(jax.jit, inline=True)
def intersection_of_rays_with_planes(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    plane_vertices: Float[ArrayLike, "*#batch 3"],
    plane_normals: Float[ArrayLike, "*#batch 3"],
) -> Float[Array, "*batch 3"]:
    """
    Return the intersection points between rays and (infinite) planes.

    Warning:
        If a ray is parallel to the corresponding plane,
        then an infinite value is returned, as a result of a division by zero,
        except for the cases where the ``ray_origins`` vertices already lie on
        the plane, then ``ray_origins`` is returned.

    Args:
        ray_origins: An array of origin vertices.
        ray_directions: An array of ray directions. The ray ends
            should be equal to ``ray_origins + ray_directions``.
        plane_vertices: An array of plane vertices. For each plane, any
            vertex on this plane can be used.
        plane_normals: an array of plane normals, where each normal has a unit
            length and is perpendicular to the corresponding plane.

    Returns:
        An array of intersection vertices.
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    plane_vertices = jnp.asarray(plane_vertices)
    plane_normals = jnp.asarray(plane_normals)

    # [*batch 3]
    u = ray_directions
    v = plane_vertices - ray_origins
    # [*batch 1]
    un = dot(u, plane_normals, keepdims=True)
    # [*batch 1]
    vn = dot(v, plane_normals, keepdims=True)

    t = vn / jnp.where(vn == 0.0, 1.0, un)
    return ray_origins + ray_directions * jnp.where(u == 0.0, 1.0, t)


@jax.jit
def image_method(
    from_vertices: Float[ArrayLike, "*#batch 3"],
    to_vertices: Float[ArrayLike, "*#batch 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
) -> Float[Array, "*batch num_mirrors 3"]:
    """
    Return the ray paths between pairs of vertices, that reflect on a given list of mirrors in between.

    The Image Method is a very simple but effective path tracing technique
    that can rapidly compute a ray path undergoing a series
    of specular reflections on a pre-defined list of mirrors.

    The method assumes infinitely long mirrors, and will return invalid
    paths in some degenerated cases such as consecutive colinear mirrors,
    or impossible configurations. It is the user's responsibility to make
    sure that the returned path is correct.

    Otherwise, the returned path will, for each reflection, have equal angles
    of incidence and of reflection.

    Warning:
        NaNs and infinity values should be treated as invalid paths, and will naturally
        occur when image paths are impossible to trace, e.g., when a mirror
        is parallel to a ray segment that it is supposed to reflect.

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

    Returns:
        An array of ray paths obtained with the image method.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`assemble_paths<differt.geometry.assemble_paths>`:

        .. code-block:: python

            paths = image_method(
                from_vertices,
                to_vertices,
                mirror_vertices,
                mirror_normals,
            )

            full_paths = assemble_paths(
                from_vertices[..., None, :],
                paths,
                to_vertices[..., None, :],
            )

    Examples:
        The following image shows how the Image Method (IM) can be applied
        to find a path between two nodes (i.e., BS and UE).

        .. figure:: ../../_static/image-method.svg
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
            intersection points, and the UE :cite:`mpt-eucap2023{fig. 5, p. 3}`.

        Next, we show how to reproduce the above results using :func:`image_method`.

        .. plotly::

            >>> from differt.geometry import TriangleMesh, normalize
            >>> from differt.plotting import draw_markers, draw_paths, reuse
            >>> from differt.rt import image_method
            >>>
            >>> from_vertex = jnp.array([+2.0, -1.0, +0.0])
            >>> to_vertex = jnp.array([+2.0, +4.0, +0.0])
            >>> mirror_vertices = jnp.array([
            ...     [3.0, 3.0, 0.0],
            ...     [4.0, 3.4, 0.0],
            ... ])
            >>> mirror_normals = jnp.array([
            ...     [+1.0, -1.0, +0.0],
            ...     [-1.0, +0.0, +0.0],
            ... ])
            >>> mirror_normals, _ = normalize(mirror_normals)
            >>> path = image_method(
            ...     from_vertex,
            ...     to_vertex,
            ...     mirror_vertices,
            ...     mirror_normals,
            ... )
            >>> with reuse(backend="plotly") as fig:  # doctest: +SKIP
            ...     TriangleMesh.plane(
            ...         mirror_vertices[0], normal=mirror_normals[0], rotate=-0.954
            ...     ).plot(color="red")
            ...     TriangleMesh.plane(
            ...         mirror_vertices[1], normal=mirror_normals[1]
            ...     ).plot(color="red")
            ...
            ...     full_path = jnp.concatenate(
            ...         (
            ...             from_vertex[None, :],
            ...             path,
            ...             to_vertex[None, :],
            ...         ),
            ...         axis=0,
            ...     )
            ...     draw_paths(full_path, marker={"color": "green"}, name="Final path")
            ...     markers = jnp.vstack((from_vertex, to_vertex))
            ...     draw_markers(
            ...         markers,
            ...         labels=["BS", "UE"],
            ...         marker={"color": "black"},
            ...         name="BS/UE",
            ...     )
            ...     fig.update_layout(scene_aspectmode="data")
            >>> fig  # doctest: +SKIP

    """
    from_vertices = jnp.asarray(from_vertices)
    to_vertices = jnp.asarray(to_vertices)
    mirror_vertices = jnp.asarray(mirror_vertices)
    mirror_normals = jnp.asarray(mirror_normals)

    # Put 'num_mirrors' axis as leading axis
    mirror_vertices = jnp.moveaxis(mirror_vertices, -2, 0)
    mirror_normals = jnp.moveaxis(mirror_normals, -2, 0)

    # Broadcast scan carries, because shapes cannot differ between input and output
    batch_and_3 = jnp.broadcast_shapes(
        from_vertices.shape,
        to_vertices.shape,
        mirror_vertices.shape[1:],
        mirror_normals.shape[1:],
    )

    from_vertices = jnp.broadcast_to(from_vertices, batch_and_3)
    to_vertices = jnp.broadcast_to(to_vertices, batch_and_3)

    def forward(
        previous_images: Float[Array, "*batch 3"],
        mirror_vertices_and_normals: tuple[
            Float[Array, "*#batch 3"], Float[Array, "*#batch 3"]
        ],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        # ruff: noqa: DOC201
        """Perform forward pass on vertices by computing consecutive images."""
        mirror_vertices, mirror_normals = mirror_vertices_and_normals
        images = image_of_vertices_with_respect_to_mirrors(
            previous_images,
            mirror_vertices,
            mirror_normals,
        )
        return images, images

    def backward(
        previous_intersections: Float[Array, "*batch 3"],
        mirror_vertices_normals_and_images: tuple[
            Float[Array, "*#batch 3"],
            Float[Array, "*#batch 3"],
            Float[Array, "*#batch 3"],
        ],
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        # ruff: noqa: DOC201
        """Perform backward pass on images by computing the intersection with mirrors."""
        mirror_vertices, mirror_normals, images = mirror_vertices_normals_and_images

        intersections = intersection_of_rays_with_planes(
            previous_intersections,
            images - previous_intersections,
            mirror_vertices,
            mirror_normals,
        )
        return intersections, intersections

    _, images = jax.lax.scan(
        forward,
        init=from_vertices,
        xs=(mirror_vertices, mirror_normals),
        unroll=True,
    )
    _, paths = jax.lax.scan(
        backward,
        init=to_vertices,
        xs=(mirror_vertices, mirror_normals, images),
        reverse=True,
        unroll=True,
    )

    return jnp.moveaxis(paths, 0, -2)  # Put 'num_mirrors' axis at the end


@jax.jit
def consecutive_vertices_are_on_same_side_of_mirrors(
    vertices: Float[ArrayLike, "*#batch num_vertices 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
) -> Bool[Array, "*#batch num_mirrors"]:
    """
    Check if consecutive vertices, but skipping one every other vertex, are on the same side of a given mirror. The number of vertices ``num_vertices`` must be equal to ``num_mirrors + 2``.

    This check is needed after using :func:`image_method` because it can return
    vertices that are behind a mirror, which causes the path to go through this
    mirror, and is something we want to avoid.

    Args:
        vertices: An array of vertices, usually describing ray paths.
        mirror_vertices: An array of mirror vertices. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normals: An array of mirror normals, where each normal has a unit
            length and is perpendicular to the corresponding mirror.

    Returns:
        A boolean array indicating whether pairs of consecutive vertices
        are on the same side of the corresponding mirror.
    """
    vertices = jnp.asarray(vertices)
    mirror_vertices = jnp.asarray(mirror_vertices)
    mirror_normals = jnp.asarray(mirror_normals)

    chex.assert_axis_dimension(
        vertices, -2, mirror_vertices.shape[-2] + 2, exception_type=TypeError
    )

    # dot_{prev,next} = <(v_{prev,next} - mirror_v), mirror_n>

    # [num_mirrors]
    d_prev = vertices[..., :-2, :] - mirror_vertices
    d_next = vertices[..., +2:, :] - mirror_vertices

    # [*batch num_mirrors]
    dot_prev = dot(d_prev, mirror_normals)
    dot_next = dot(d_next, mirror_normals)

    return jnp.sign(dot_prev) == jnp.sign(dot_next)

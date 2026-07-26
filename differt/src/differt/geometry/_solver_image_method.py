from typing import overload

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from differt.utils import smoothing_function


@jax.jit(inline=True)
def image_of_vertex_with_respect_to_mirror(
    vertex: Float[ArrayLike, "*#batch 3"],
    mirror_vertex: Float[ArrayLike, "*#batch 3"],
    mirror_normal: Float[ArrayLike, "*#batch 3"],
) -> Float[Array, "*batch 3"]:
    """
    Return the image of the vertex with respect to the mirror.

    Args:
        vertex: Vertex that will be mirrored.
        mirror_vertex: Mirror vertex. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normal: Mirror normal, where each normal has a unit
            length and is perpendicular to the corresponding mirror.

    Returns:
        The image of the vertex.

    Examples:
        In the following example, we show how to compute the images of
        a batch of random vertices. Here, normal vectors do not have a unit length,
        but they should have if you want an interpretable result.

        >>> from differt.rt import (
        ...     image_of_vertex_with_respect_to_mirror,
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
        >>> images = image_of_vertex_with_respect_to_mirror(
        ...     vertices,
        ...     mirror_vertices,
        ...     mirror_normals,
        ... )
        >>> images.shape
        (10, 20, 30, 3)

    """
    vertex = jnp.asarray(vertex)
    mirror_vertex = jnp.asarray(mirror_vertex)
    mirror_normal = jnp.asarray(mirror_normal)

    # [*batch num_mirrors ]
    incident = vertex - mirror_vertex  # incident vectors
    return (
        vertex
        - 2.0
        * jnp.sum(incident * mirror_normal, axis=-1, keepdims=True)
        * mirror_normal
    )


@jax.jit(inline=True)
def intersection_of_ray_with_plane(
    ray_origin: Float[ArrayLike, "*#batch 3"],
    ray_direction: Float[ArrayLike, "*#batch 3"],
    plane_vertex: Float[ArrayLike, "*#batch 3"],
    plane_normal: Float[ArrayLike, "*#batch 3"],
) -> Float[Array, "*batch 3"]:
    """
    Return the intersection point between the ray and the (infinite) plane.

    Warning:
        If a ray is parallel to the corresponding plane,
        then an infinite value is returned, as a result of a division by zero,
        except for the cases where the ``ray_origins`` vertices already lie on
        the plane, then ``ray_origins`` is returned.

    Args:
        ray_origin: Origin vertex.
        ray_direction: Ray direction. The ray end
            should be equal to ``ray_origin + ray_direction``.
        plane_vertex: Plane vertex. For each plane, any
            vertex on this plane can be used.
        plane_normal: Plane normal, where each normal has a unit
            length and is perpendicular to the corresponding plane.

    Returns:
        Intersection point with the (infinite) plane.
    """
    ray_origin = jnp.asarray(ray_origin)
    ray_direction = jnp.asarray(ray_direction)
    plane_vertex = jnp.asarray(plane_vertex)
    plane_normal = jnp.asarray(plane_normal)

    # [*batch 3]
    u = ray_direction
    v = plane_vertex - ray_origin
    # [*batch 1]
    un = jnp.sum(u * plane_normal, axis=-1, keepdims=True)
    # [*batch 1]
    vn = jnp.sum(v * plane_normal, axis=-1, keepdims=True)

    parallel = un == 0.0
    un = jnp.where(parallel, jnp.ones_like(un), un)

    t = vn / un

    shape = jnp.broadcast_shapes(ray_origin.shape, ray_direction.shape, t.shape)
    dtype = jnp.result_type(ray_origin, ray_direction, t)

    return jnp.where(
        parallel & (vn != 0.0),
        jnp.full(shape, jnp.inf, dtype=dtype),
        ray_origin + ray_direction * t,
    )


def _forward(
    previous_image: Float[Array, "3"],
    mirror_vertex_and_normal: tuple[Float[Array, "3"], Float[Array, "3"]],
) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
    # Perform forward pass on vertices by computing consecutive images.
    mirror_vertex, mirror_normal = mirror_vertex_and_normal
    image = image_of_vertex_with_respect_to_mirror(
        previous_image,
        mirror_vertex,
        mirror_normal,
    )
    return image, image


def _backward(
    previous_intersection: Float[Array, "3"],
    mirror_vertex_normal_and_image: tuple[
        Float[Array, "3"],
        Float[Array, "3"],
        Float[Array, "3"],
    ],
) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
    # Perform backward pass on images by computing the intersection with mirrors.
    mirror_vertex, mirror_normal, image = mirror_vertex_normal_and_image

    # We avoid NaNs (caused by subtraction of two infinities) by replacing
    # previous_intersections with zeros when they are infinite.
    no_previous_intersection = jnp.isinf(previous_intersection)
    previous_intersection = jnp.where(
        no_previous_intersection,
        jnp.zeros_like(previous_intersection),
        previous_intersection,
    )
    intersection = intersection_of_ray_with_plane(
        previous_intersection,
        image - previous_intersection,
        mirror_vertex,
        mirror_normal,
    )
    intersection: Array = jnp.where(
        no_previous_intersection,
        jnp.full_like(intersection, jnp.inf),
        intersection,
    )
    return intersection, intersection


def _image_method(
    from_vertex: Float[Array, "3"],
    to_vertex: Float[Array, "3"],
    mirror_vertices: Float[Array, "num_mirrors 3"],
    mirror_normals: Float[Array, "num_mirrors 3"],
) -> Float[Array, "num_mirrors 3"]:
    _, images = jax.lax.scan(
        _forward,
        init=from_vertex,
        xs=(mirror_vertices, mirror_normals),
    )
    _, paths = jax.lax.scan(
        _backward,
        init=to_vertex,
        xs=(mirror_vertices, mirror_normals, images),
        reverse=True,
    )

    return paths


@jax.jit
def image_method(
    from_vertex: Float[ArrayLike, "*#batch 3"],
    to_vertex: Float[ArrayLike, "*#batch 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
) -> Float[Array, "*batch num_mirrors 3"]:
    """
    Return the ray path between a pair of vertices, that reflects on a given list of mirrors in between.

    The Image Method is a very simple but effective path tracing technique
    that can rapidly compute a ray path undergoing a series
    of specular reflections on a pre-defined list of mirrors.

    The method assumes infinitely long mirrors, and will return invalid
    paths in some degenerated cases such as consecutive collinear mirrors,
    or impossible configurations. It is the user's responsibility to make
    sure that the returned path is correct.

    Otherwise, the returned path will, for each reflection, have equal angles
    of incidence and of reflection.

    Warning:
        NaNs and infinity values should be treated as invalid paths, and will naturally
        occur when image paths are impossible to trace, e.g., when a mirror
        is parallel to a ray segment that it is supposed to reflect.

    Args:
        from_vertex: ``from`` vertex, i.e., vertex from which the
            ray path starts. In a radio communications context, this is usually
            the transmitter position.
        to_vertex: ``to`` vertex, i.e., vertex to which the
            ray path ends. In a radio communications context, this is usually
            the receiver position.
        mirror_vertices: Mirror vertex. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normals: Mirror normal, where each normal has a unit
            length and is perpendicular to the corresponding mirror.

    Returns:
        Intermediate ray path vertices obtained with the image method.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`assemble_path<differt.geometry.assemble_path>`:

        .. code-block:: python

            path = image_method(
                from_vertex,
                to_vertex,
                mirror_vertices,
                mirror_normals,
            )

            full_path = assemble_path(
                from_vertex,
                path,
                to_vertex,
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
            intersections with mirrors are computed backward, i.e., from last mirror to
            first, by joining the UE, then the intersection points, with the images of the
            BS. Finally, the valid path can be obtained by joining BS, the intermediary
            intersection points, and the UE :cite:`mpt-eucap2023{fig. 5, p. 3}`.

        Next, we show how to reproduce the above results using :func:`image_method`.

        .. plotly::

            >>> from differt.geometry import Mesh, normalize, assemble_path
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
            ...     Mesh.plane(
            ...         mirror_vertices[0], normal=mirror_normals[0], rotate=-0.954
            ...     ).plot(color="red")
            ...     Mesh.plane(mirror_vertices[1], normal=mirror_normals[1]).plot(
            ...         color="red"
            ...     )
            ...
            ...     full_path = assemble_path(
            ...         from_vertex,
            ...         path,
            ...         to_vertex,
            ...     )
            ...     draw_paths(
            ...         full_path,
            ...         mode="lines+markers",
            ...         marker={"color": "green"},
            ...         name="Final path",
            ...     )
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
    from_vertex = jnp.asarray(from_vertex)
    to_vertex = jnp.asarray(to_vertex)
    mirror_vertices = jnp.asarray(mirror_vertices)
    mirror_normals = jnp.asarray(mirror_normals)

    if mirror_vertices.shape[-2] == 0:
        # If there are no mirrors, return empty array.
        batch = jnp.broadcast_shapes(
            from_vertex.shape[:-1],
            to_vertex.shape[:-1],
            mirror_vertices.shape[:-2],
            mirror_normals.shape[:-2],
        )
        dtype = jnp.result_type(from_vertex, to_vertex, mirror_vertices, mirror_normals)
        return jnp.empty((*batch, 0, 3), dtype=dtype)

    return jnp.vectorize(
        _image_method,
        signature="(3),(3),(n,3),(n,3)->(n,3)",
    )(from_vertex, to_vertex, mirror_vertices, mirror_normals)


@overload
def consecutive_vertices_are_on_same_side_of_mirror(
    vertices: Float[ArrayLike, "*#batch num_vertices 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
    *,
    smoothing_factor: None = ...,
) -> Bool[Array, "*#batch num_mirrors"]: ...


@overload
def consecutive_vertices_are_on_same_side_of_mirror(
    vertices: Float[ArrayLike, "*#batch num_vertices 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
    *,
    smoothing_factor: Float[ArrayLike, ""],
) -> Float[Array, "*#batch num_mirrors"]: ...


@jax.jit
def consecutive_vertices_are_on_same_side_of_mirror(
    vertices: Float[ArrayLike, "*#batch num_vertices 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
    *,
    smoothing_factor: Float[ArrayLike, ""] | None = None,
) -> Bool[Array, "*#batch num_mirrors"] | Float[Array, "*#batch num_mirrors"]:
    """
    Check if consecutive vertices, but skipping one every other vertex, are on the same side of a given mirror. The number of vertices ``num_vertices`` must be equal to ``num_mirrors + 2``.

    This check is needed after using :func:`image_method` because it can return
    vertices that are behind a mirror, which causes the path to go through this
    mirror, and is something we want to avoid.

    Args:
        vertices: Vertices, usually describing the ray path.
        mirror_vertices: Mirror vertices. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normals: Mirror normals, where each normal has a unit
            length and is perpendicular to the corresponding mirror.
        smoothing_factor: If set, hard conditions are replaced with smoothed ones,
            as described in :cite:`fully-eucap2024`, and this argument parameters the slope
            of the smoothing function. The output value is now a real value
            between 0 (:data:`False`) and 1 (:data:`True`).

            For more details, refer to :ref:`smoothing`.

    Returns:
        Boolean mask, ``True`` if both vertices on either side of each mirror are on the same side.
    """
    vertices = jnp.asarray(vertices)
    mirror_vertices = jnp.asarray(mirror_vertices)
    mirror_normals = jnp.asarray(mirror_normals)

    chex.assert_axis_dimension(
        vertices, -2, mirror_vertices.shape[-2] + 2, exception_type=TypeError
    )

    if mirror_vertices.shape[-2] == 0:
        # If there are no mirrors, return empty array.
        batch = jnp.broadcast_shapes(
            vertices.shape[:-2],
            mirror_vertices.shape[:-2],
            mirror_normals.shape[:-2],
        )
        dtype = (
            bool
            if smoothing_factor is None
            else jnp.result_type(vertices, mirror_vertices, mirror_normals)
        )
        return jnp.empty((*batch, 0), dtype=dtype)

    # dot_{prev,next} = <(v_{prev,next} - mirror_v), mirror_n>

    # [*batch num_mirrors 3]
    d_prev = vertices[..., :-2, :] - mirror_vertices
    d_next = vertices[..., +2:, :] - mirror_vertices

    # [*batch num_mirrors]
    dot_prev = jnp.sum(d_prev * mirror_normals, axis=-1)
    dot_next = jnp.sum(d_next * mirror_normals, axis=-1)

    if smoothing_factor is not None:
        return smoothing_function(
            jnp.sign(dot_prev) * jnp.sign(dot_next), smoothing_factor
        )
    return jnp.sign(dot_prev) == jnp.sign(dot_next)

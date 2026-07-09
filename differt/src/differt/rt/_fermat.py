from typing import Any, no_type_check

import equinox as eqx
import fpt_jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from differt.geometry import orthogonal_basis


@eqx.filter_jit
def fermat_path_on_linear_objects(
    from_vertex: Float[ArrayLike, "*#batch 3"],
    to_vertex: Float[ArrayLike, "*#batch 3"],
    object_origins: Float[ArrayLike, "*#batch num_objects 3"],
    object_vectors: Float[ArrayLike, "*#batch num_objects num_dims 3"],
    *,
    steps: int = 10,
    unroll: int | bool = 1,
    linesearch_steps: int = 1,
    unroll_linesearch: int | bool = 1,
    implicit_diff: bool = True,
) -> Float[Array, "*batch num_objects 3"]:
    """
    Return the ray path between a pair of vertices, that reflects or diffracts on a given list of objects in between.

    Linear objects are defined by a linear combination of zero or more (possibly orthogonal) vectors,
    plus a point in space (i.e., the origin).

    E.g., a straight line can be defined by a point and a direction vector,
    while a plane can be defined by a point and two orthogonal direction vectors.

    Because the size of ``object_vectors`` is determined by the object with the most dimensions,
    objects with fewer dimensions should have the extra vectors set to zero.

    Based on the Fermat principle, this method finds the ray path
    between the ``from`` and ``to`` vertices, that minimizes the total length of the path.

    While the method assumes that the objects are infinite, as for the
    :func:`image method<differt.rt.image_method>`, choosing an appropriate origin can be important
    as it will be used as the initial point of the minimization procedure.

    .. important::

        The current implementation uses the ``fpt-jax`` library
        :cite:`fpt-eucap2026`, which defines :class:`jax.custom_vjp` functions for more
        efficient gradient computations, see the paper for more details. The implementation
        is not guaranteed to converge for all configurations of vertices and objects,
        and the user may need to tune the number of steps to achieve convergence.

    Args:
        from_vertex: ``from`` vertex, i.e., vertex from which the
            ray path starts. In a radio communications context, this is usually
            the transmitter position.
        to_vertex: ``to`` vertex, i.e., vertex to which the
            ray path ends. In a radio communications context, this is usually
            the receiver position.
        object_origins: Object origins, used as the initial guess of the minimization procedure.
        object_vectors: Base vector(s) describing each object.
        steps: The number of optimization steps to perform.
        unroll: Whether to unroll the optimization loop. Can be a boolean or an integer
            specifying the number of iterations to unroll, see :func:`jax.lax.scan`.
        linesearch_steps: The number of line search steps to perform at each iteration.
        unroll_linesearch: Whether to unroll the line search loop. Can be a boolean or an integer
            specifying the number of iterations to unroll, see :func:`jax.lax.scan`.
        implicit_diff: Whether to use implicit differentiation for computing the gradient.
            See :cite:`fpt-eucap2026` and its
            `GitHub page <https://github.com/jeertmans/fpt-jax/tree/v0.1.0>`_ for more details.

    Returns:
        Intermediate ray path vertices obtained using Fermat's principle.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`assemble_path<differt.geometry.assemble_path>`:

        .. code-block:: python

            path = fermat_path_on_linear_objects(...)

            full_path = assemble_path(
                from_vertex,
                path,
                to_vertex,
            )

    Examples:
        The following example shows how to use this method to find the ray paths
        undergoing a diffraction on the edge of a wall, then a reflection on a mirror,
        before reaching the receiver.

        .. plotly::

            >>> from differt.geometry import TriangleMesh, normalize, assemble_path
            >>> from differt.plotting import draw_markers, draw_paths, reuse
            >>> from differt.rt import fermat_path_on_linear_objects
            >>>
            >>> from_vertex = jnp.array([-2.0, 0.0, 0.0])
            >>> to_vertex = jnp.array([0.0, 0.0, 0.0])
            >>> wall = TriangleMesh.plane(
            ...     jnp.array([-1.0, 0.0, 0.0]), normal=jnp.array([1.0, 0.0, 0.0])
            ... )
            >>> mirror = TriangleMesh.plane(
            ...     jnp.array([1.0, 0.0, 0.0]), normal=jnp.array([1.0, 0.0, 0.0])
            ... )
            >>> object_origins = jnp.array([[-1.0, -0.5, 0.5], [1.0, 0.0, 0.0]])
            >>> object_vectors = jnp.array([
            ...     [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],  # Edges only need one vector
            ...     [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Planes need two vectors
            ... ])
            >>> path = fermat_path_on_linear_objects(
            ...     from_vertex,
            ...     to_vertex,
            ...     object_origins,
            ...     object_vectors,
            ... )
            >>> with reuse(backend="plotly") as fig:  # doctest: +SKIP
            ...     wall.plot(color="blue")
            ...     mirror.plot(color="red")
            ...     draw_paths(
            ...         wall.triangle_edges[0, 1, ...],
            ...         line_color="yellow",
            ...         line_width=5,
            ...         name="Edge",
            ...     )
            ...
            ...     full_path = assemble_path(
            ...         from_vertex,
            ...         path,
            ...         to_vertex,
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
    from_vertex = jnp.asarray(from_vertex)
    to_vertex = jnp.asarray(to_vertex)
    object_origins = jnp.asarray(object_origins)
    object_vectors = jnp.asarray(object_vectors)

    if (num_objects := object_origins.shape[-2]) == 0 or (
        object_vectors.shape[-2] == 0
    ):
        # If there are no objects, return empty paths.
        # If there are no dimensions, return origins.
        batch = jnp.broadcast_shapes(
            from_vertex.shape[:-1],
            to_vertex.shape[:-1],
            object_origins.shape[:-2],
            object_vectors.shape[:-3],
        )
        dtype = jnp.result_type(from_vertex, to_vertex, object_origins, object_vectors)
        if num_objects == 0:
            return jnp.empty((*batch, 0, 3), dtype=dtype)
        return jnp.broadcast_to(object_origins, (*batch, num_objects, 3)).astype(dtype)

    # Needed until https://github.com/jax-ml/jax/issues/34697 is resolved
    @no_type_check
    def _call_fpt_jax() -> Array:
        return fpt_jax.trace_rays(
            from_vertex,
            to_vertex,
            object_origins,
            object_vectors,
            num_iters=steps,
            unroll=unroll,
            num_iters_linesearch=linesearch_steps,
            unroll_linesearch=unroll_linesearch,
            implicit_diff=implicit_diff,
        )

    return _call_fpt_jax()


@eqx.filter_jit
def fermat_path_on_planar_mirrors(
    from_vertex: Float[ArrayLike, "*#batch 3"],
    to_vertex: Float[ArrayLike, "*#batch 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
    **kwargs: Any,
) -> Float[Array, "*batch num_mirrors 3"]:
    """
    Return the ray path between a pair of vertices, that reflects on a given list of mirrors in between.

    Args:
        from_vertex: ``from`` vertex, i.e., vertex from which the
            ray path starts. In a radio communications context, this is usually
            the transmitter position.
        to_vertex: ``to`` vertex, i.e., vertex to which the
            ray path ends. In a radio communications context, this is usually
            the receiver position.
        mirror_vertices: Mirror vertices. For each mirror, any
            vertex on the infinite plane that describes the mirror is considered
            to be a valid vertex.
        mirror_normals: Mirror normals, where each normal has a unit
            length and is perpendicular to the corresponding mirror.

            .. note::

                Unlike with the Image method, the normals do not actually have to
                be unit vectors. However, we keep the same documentation so it is
                easier for the user to move from one method to the other.
        kwargs: Keyword arguments passed to
            :func:`fermat_path_on_linear_objects`.

    Returns:
        Intermediate ray path vertices obtained using Fermat's principle.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`assemble_path<differt.geometry.assemble_path>`:

        .. code-block:: python

            path = fermat_path_on_planar_mirrors(...)

            full_path = assemble_path(
                from_vertex,
                path,
                to_vertex,
            )

    Examples:
        The following example is the same as for the
        :func:`image_method<differt.rt.image_method>`, but using the Fermat principle.

        .. plotly::

            >>> from differt.geometry import TriangleMesh, normalize, assemble_path
            >>> from differt.plotting import draw_markers, draw_paths, reuse
            >>> from differt.rt import fermat_path_on_planar_mirrors
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
            >>> path = fermat_path_on_planar_mirrors(
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
            ...     full_path = assemble_path(
            ...         from_vertex,
            ...         path,
            ...         to_vertex,
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
    mirror_directions_1, mirror_directions_2 = orthogonal_basis(mirror_normals)

    object_origins = mirror_vertices
    object_vectors = jnp.stack(
        (
            mirror_directions_1,
            mirror_directions_2,
        ),
        axis=-2,
    )

    return fermat_path_on_linear_objects(
        from_vertex, to_vertex, object_origins, object_vectors, **kwargs
    )

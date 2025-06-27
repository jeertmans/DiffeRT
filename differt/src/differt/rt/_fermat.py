from functools import partial
from typing import Any, no_type_check

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Float

from differt.geometry import orthogonal_basis


@partial(jax.jit, inline=True)
def _param_to_xyz(
    p: Float[Array, "*batch num_object_x_num_dims"],
    o: Float[Array, "*#batch num_objects 3"],
    v: Float[Array, "*#batch num_objects num_dims 3"],
) -> Float[Array, "*batch num_objects 3"]:
    *_, num_objects, num_dims, _ = v.shape
    p = jnp.reshape(p, (*p.shape[:-1], num_objects, num_dims, 1))
    return o + jnp.sum(p * v, axis=-2)


@partial(jax.jit, inline=True)
def _loss(
    p: Float[Array, "*batch num_object_x_num_dims"],
    from_: Float[Array, "*#batch 3"],
    to: Float[Array, "*#batch 3"],
    o: Float[Array, "*#batch num_objects 3"],
    v: Float[Array, "*#batch num_objects num_dims 3"],
) -> Float[Array, " *batch"]:
    chex.assert_axis_dimension_gt(p, -1, 0, exception_type=TypeError)
    chex.assert_axis_dimension(
        p, -1, v.shape[-2] * v.shape[-3], exception_type=TypeError
    )
    xyz = _param_to_xyz(p, o, v)
    first_segment = xyz[..., 0, :] - from_
    last_segment = to - xyz[..., -1, :]
    other_segments = jnp.diff(xyz, axis=-2)
    return (
        jnp.linalg.norm(first_segment, axis=-1)
        + jnp.linalg.norm(last_segment, axis=-1)
        + jnp.sum(jnp.linalg.norm(other_segments, axis=-1), axis=-1)
    )


@eqx.filter_jit
def fermat_path_on_linear_objects(
    from_vertices: Float[ArrayLike, "*#batch 3"],
    to_vertices: Float[ArrayLike, "*#batch 3"],
    object_origins: Float[ArrayLike, "*#batch num_objects 3"],
    object_vectors: Float[ArrayLike, "*#batch num_objects num_dims 3"],
    *,
    steps: int = 10,
    optimizer: optax.GradientTransformationExtraArgs | None = None,
) -> Float[Array, "*batch num_objects 3"]:
    """
    Return the ray paths between pairs of vertices, that reflect or diffract on a given list of objects in between.

    Linear objects are defined by a linear combination of zero or more (possibly orthogonal) vectors,
    plus a point in space (i.e., the origin).

    E.g., a straight line can be defined by a point and a direction vector,
    while a plane can be defined by a point and two orthogonal direction vectors.

    Because the size of ``object_vectors`` is determined by the object with the most dimensions,
    objects with fewer dimensions should have the extra vectors set to zero.

    Based on the Fermat principle, this method finds the ray paths
    between the ``from`` and ``to`` vertices, that minimize the total length of the paths.

    While the method assumes that the objects are infinite, as for the
    :func:`image method<differt.rt.image_method>`, choosing an appropriate origin can be important
    as it will be used as the initial point of the minimization procedure.

    Args:
        from_vertices: An array of ``from`` vertices, i.e., vertices from which the
            ray paths start. In a radio communications context, this is usually
            an array of transmitters.
        to_vertices: An array of ``to`` vertices, i.e., vertices to which the
            ray paths end. In a radio communications context, this is usually
            an array of receivers.
        object_origins: An array of object origins.

            It is used as the initial guess of the minimization procedure.
        object_vectors: An array of base vectors describing the objects.
        steps: The number of optimization steps to perform.
        optimizer: The optimizer to use. If not provided,
            uses :func:`optax.lbfgs`.

            .. important::

                The optimizer should store the gradient in the state. In the future,
                we hope to support optimizers that do not store the gradient in the state,
                such as :func:`optax.adam` or :func:`optax.sgd`.

    Returns:
        An array of ray paths obtained based on Fermat's principle.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`assemble_paths<differt.geometry.assemble_paths>`:

        .. code-block:: python

            paths = fermat_path_on_linear_objects(...)

            full_paths = assemble_paths(
                from_vertices[..., None, :],
                paths,
                to_vertices[..., None, :],
            )

    Examples:
        The following example shows how to use this method to find the ray paths
        undergoing a diffraction on the edge of a wall, then a reflection on a mirror,
        before reaching the receiver.

        .. plotly::

            >>> from differt.geometry import TriangleMesh, normalize, assemble_paths
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
            ...     full_path = assemble_paths(
            ...         from_vertex[None, :],
            ...         path,
            ...         to_vertex[None, :],
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
    object_origins = jnp.asarray(object_origins)
    object_vectors = jnp.asarray(object_vectors)

    batch = jnp.broadcast_shapes(
        from_vertices.shape[:-1],
        to_vertices.shape[:-1],
        object_origins.shape[:-2],
        object_vectors.shape[:-3],
    )

    num_objects = object_origins.shape[-2]

    if num_objects == 0:
        # If there are no objects, return empty paths.
        dtype = jnp.result_type(
            from_vertices, to_vertices, object_origins, object_vectors
        )
        return jnp.empty((*batch, 0, 3), dtype=dtype)

    num_dims = object_vectors.shape[-2]

    if num_dims == 0:
        # If there are no dimension, return origins.
        dtype = jnp.result_type(
            from_vertices, to_vertices, object_origins, object_vectors
        )
        return jnp.broadcast_to(object_origins, (*batch, num_objects, 3)).astype(dtype)

    num_unknowns = num_objects * num_dims

    objective = _loss

    optimizer = optimizer if optimizer is not None else optax.lbfgs()

    def minimize(
        x_0: Float[Array, " num_unknowns"],
        from_: Float[Array, "3"],
        to: Float[Array, "3"],
        o: Float[Array, "num_objects 3"],
        v: Float[Array, "num_objects num_dims 3"],
    ) -> Float[Array, "num_objects 3"]:
        opt_state = optimizer.init(x_0)

        value_and_grad = optax.value_and_grad_from_state(objective)

        @no_type_check
        def update(
            state: tuple[Float[Array, "*batch num_unknowns"], Any],
            _: None,
        ) -> tuple[
            tuple[Float[Array, "*batch num_unknowns"], Any], float | Float[Array, " "]
        ]:
            x, opt_state = state
            loss_value, grad = value_and_grad(x, from_, to, o, v, state=opt_state)
            updates, opt_state = optimizer.update(
                grad,
                opt_state,
                x,
                value=loss_value,
                grad=grad,
                value_fn=objective,
                from_=from_,
                to=to,
                o=o,
                v=v,
            )
            x = optax.apply_updates(x, updates)
            return (x, opt_state), loss_value

        (x, _), _ = jax.lax.scan(update, (x_0, opt_state), length=steps)
        return _param_to_xyz(x, o, v)

    x_0 = jnp.zeros((*batch, num_unknowns))

    for i in reversed(range(len(batch))):
        minimize = jax.vmap(
            minimize,
            in_axes=(
                0,
                0 if from_vertices.ndim > i + 1 else None,
                0 if to_vertices.ndim > i + 1 else None,
                0 if object_origins.ndim > i + 2 else None,
                0 if object_vectors.ndim > i + 3 else None,
            ),
        )

    return minimize(x_0, from_vertices, to_vertices, object_origins, object_vectors)


@eqx.filter_jit
def fermat_path_on_planar_mirrors(
    from_vertices: Float[ArrayLike, "*#batch 3"],
    to_vertices: Float[ArrayLike, "*#batch 3"],
    mirror_vertices: Float[ArrayLike, "*#batch num_mirrors 3"],
    mirror_normals: Float[ArrayLike, "*#batch num_mirrors 3"],
    **kwargs: Any,
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
            length and is perpendicular to the corresponding mirror.

            .. note::

                Unlike with the Image method, the normals do not actually have to
                be unit vectors. However, we keep the same documentation so it is
                easier for the user to move from one method to the other.
        kwargs: Keyword arguments passed to
            :func:`fermat_path_on_linear_objects`.

    Returns:
        An array of ray paths obtained using Fermat's principle.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`assemble_paths<differt.geometry.assemble_paths>`:

        .. code-block:: python

            paths = fermat_path_on_planar_mirrors(...)

            full_paths = assemble_paths(
                from_vertices[..., None, :],
                paths,
                to_vertices[..., None, :],
            )

    Examples:
        The following example is the same as for the
        :func:`image_method<differt.rt.image_method>`, but using the Fermat principle.

        .. plotly::

            >>> from differt.geometry import TriangleMesh, normalize, assemble_paths
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
            ...     full_path = assemble_paths(
            ...         from_vertex[None, :],
            ...         path,
            ...         to_vertex[None, :],
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
        from_vertices, to_vertices, object_origins, object_vectors, **kwargs
    )

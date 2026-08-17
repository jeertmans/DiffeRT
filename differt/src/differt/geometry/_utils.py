import typing
from collections.abc import Callable, Iterator, Sized
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, TypeVar, no_type_check, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, DTypeLike, Float, Int

from differt.utils import smoothing_function
from differt_core.geometry import CompleteGraph


@overload
def normalize(
    vectors: Float[ArrayLike, "*batch 3"],
    keepdims: Literal[False] = False,
) -> tuple[Float[Array, "*batch 3"], Float[Array, " *batch"]]: ...


@overload
def normalize(
    vectors: Float[ArrayLike, "*batch 3"],
    keepdims: Literal[True],
) -> tuple[Float[Array, "*batch 3"], Float[Array, " *batch 1"]]: ...


@jax.jit(static_argnames=("keepdims",), inline=True)
def normalize(
    vectors: Float[ArrayLike, "*batch 3"],
    keepdims: bool = False,
) -> tuple[
    Float[Array, "*batch 3"], Float[Array, " *batch"] | Float[Array, " *batch 1"]
]:
    """
    Normalize vectors and also return their length.

    This function avoids division by zero by checking vectors
    with zero-length, dividing by one instead.

    Args:
        vectors: Input vector.
        keepdims: If set to :data:`True`, the array of lengths
            will have the same number of dimensions as the input.

    Returns:
        The normalized vector and its length.

    Examples:
        The following examples shows how normalization works and
        its special behavior at zero.

        >>> from differt.geometry import (
        ...     normalize,
        ... )
        >>>
        >>> vector = jnp.array([1.0, 1.0, 1.0])
        >>> normalize(vector)  # [1., 1., 1.] / sqrt(3), sqrt(3)
        (Array([0.5773503, 0.5773503, 0.5773503], dtype=float32),
         Array(1.7320508, dtype=float32))
        >>> zero = jnp.array([0.0, 0.0, 0.0])
        >>> normalize(zero)  # Special behavior at 0.
        (Array([0., 0., 0.], dtype=float32), Array(0., dtype=float32))
    """
    vectors = jnp.asarray(vectors)
    lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    lengths_no_zero: Array = jnp.where(lengths == 0.0, jnp.ones_like(lengths), lengths)

    return vectors / lengths_no_zero, (
        lengths if keepdims else jnp.squeeze(lengths, axis=-1)
    )


@jax.jit(inline=True)
def perpendicular_vector(u: Float[ArrayLike, "*batch 3"]) -> Float[Array, "*batch 3"]:
    """
    Generate a vector perpendicular to the input vector.

    Args:
        u: The input vector.

    Returns:
        Vector perpendicular to the input vector.

    Examples:
        The following example shows how this function works on basic input vectors.

        >>> from differt.geometry import (
        ...     perpendicular_vector,
        ... )
        >>>
        >>> u = jnp.array([1.0, 0.0, 0.0])
        >>> perpendicular_vector(u)
        Array([ 0., -0.,  1.], dtype=float32)
        >>> u = jnp.array([1.0, 1.0, 1.0])
        >>> perpendicular_vector(u)
        Array([ 0.8164966, -0.4082483, -0.4082483], dtype=float32)
    """
    u = jnp.asarray(u)
    z = jnp.zeros_like(u[..., 0])
    v = jnp.where(
        (jnp.abs(u[..., 0]) > jnp.abs(u[..., 1]))[..., None],
        jnp.stack((-u[..., 1], u[..., 0], z), axis=-1),
        jnp.stack((z, -u[..., 2], u[..., 1]), axis=-1),
    )
    w = jnp.cross(u, v)
    return normalize(w)[0]


@jax.jit(inline=True)
def orthogonal_basis(
    u: Float[ArrayLike, "*batch 3"],
) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """
    Generate ``v`` and ``w``, two other unit vectors that form with input ``u`` an orthogonal basis.

    Args:
        u: The first direction of the orthogonal basis.
            It must have a unit length.

    Returns:
        A pair of unit vectors, ``v`` and ``w``.

    Examples:
        The following example shows how this function works on basic input vectors.

        >>> from differt.geometry import (
        ...     normalize,
        ...     orthogonal_basis,
        ... )
        >>>
        >>> u = jnp.array([1.0, 0.0, 0.0])
        >>> orthogonal_basis(u)
        (Array([-0., 1.,  0.], dtype=float32), Array([ 0., -0., 1.], dtype=float32))
        >>> u, _ = normalize(jnp.array([1.0, 1.0, 1.0]))
        >>> orthogonal_basis(u)
        (Array([-0.       , -0.7071068,  0.7071068], dtype=float32),
         Array([ 0.8164966, -0.4082483, -0.4082483], dtype=float32))
    """
    u = jnp.asarray(u)
    w = perpendicular_vector(u)
    v = jnp.cross(w, u)
    v = normalize(v)[0]

    return v, w


@jax.jit(inline=True)
def path_length(
    path: Float[ArrayLike, "*batch path_length 3"],
) -> Float[Array, " *batch"]:
    """
    Compute the path length of the path.

    The path is exactly made of ``path_length`` vertices.

    Args:
        path: Input path.

    Returns:
        The path length.

    Examples:
        The following example shows how to compute the length of a very simple path.

        >>> from differt.geometry import (
        ...     path_length,
        ... )
        >>>
        >>> path = jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        >>> path_length(path)
        Array(1., dtype=float32)
        >>> path_length(jnp.vstack((path, path[::-1, :])))
        Array(2., dtype=float32)
    """
    path = jnp.asarray(path)
    vectors = jnp.diff(path, axis=-2)
    lengths = jnp.linalg.norm(vectors, axis=-1)

    return jnp.sum(lengths, axis=-1)


@jax.jit
def rotation_matrix_along_x_axis(
    angle: Float[ArrayLike, ""],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along x axis.

    Args:
        angle: The rotation angle, in radians.

    Returns:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along the x axis.

        >>> from differt.geometry import (
        ...     rotation_matrix_along_x_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 1.0, 0.0])
        >>> rotation_matrix_along_x_axis(jnp.pi / 2) @ xyz
        Array([ 1., -0.,  1.], dtype=float32)
    """
    co = jnp.cos(angle)
    si = jnp.sin(angle)

    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, +co, -si],
        [0.0, +si, +co],
    ])


@jax.jit
def rotation_matrix_along_y_axis(
    angle: Float[ArrayLike, ""],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along y axis.

    Args:
        angle: The rotation angle, in radians.

    Returns:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along the y axis.

        >>> from differt.geometry import (
        ...     rotation_matrix_along_y_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 1.0, 0.0])
        >>> rotation_matrix_along_y_axis(jnp.pi / 2) @ xyz
        Array([-0.,  1., -1.], dtype=float32)
    """
    co = jnp.cos(angle)
    si = jnp.sin(angle)

    return jnp.array([
        [+co, 0.0, +si],
        [0.0, 1.0, 0.0],
        [-si, 0.0, +co],
    ])


@jax.jit
def rotation_matrix_along_z_axis(
    angle: Float[ArrayLike, ""],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along z axis.

    Args:
        angle: The rotation angle, in radians.

    Returns:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along the z axis.

        >>> from differt.geometry import (
        ...     rotation_matrix_along_z_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 0.0, 1.0])
        >>> rotation_matrix_along_z_axis(jnp.pi / 2) @ xyz
        Array([-0.,  1.,  1.], dtype=float32)
    """
    co = jnp.cos(angle)
    si = jnp.sin(angle)

    return jnp.array([
        [+co, -si, 0.0],
        [+si, +co, 0.0],
        [0.0, 0.0, 1.0],
    ])


@jax.jit
def rotation_matrix_along_axis(
    angle: Float[ArrayLike, ""],
    axis: Float[ArrayLike, "3"],
) -> Float[Array, "3 3"]:
    """
    Return a rotation matrix to rotate coordinates along a given axis.

    Args:
        angle: The rotation angle, in radians.
        axis: A unit vector pointing in the axis' direction.

    Returns:
        The rotation matrix.

    Examples:
        The following example shows how to rotate xyz coordinates
        along a given axis.

        >>> from differt.geometry import (
        ...     normalize,
        ...     rotation_matrix_along_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 1.0, 1.0])
        >>> axis, _ = normalize(jnp.array([1.0, 1.0, 0.0]))
        >>> rotation_matrix_along_axis(jnp.pi / 2, axis) @ xyz
        Array([ 1.7071066,  0.2928931, -0.       ], dtype=float32)

        In the following example, we show the importance of using a unit
        vector.

        >>> from differt.geometry import (
        ...     rotation_matrix_along_axis,
        ... )
        >>>
        >>> xyz = jnp.array([1.0, 0.0, 1.0])
        >>> axis = jnp.array([1.0, 0.0, 0.0])
        >>> rotation_matrix_along_axis(jnp.pi, axis) @ xyz
        Array([ 1.       ,  0.0000001, -1.       ], dtype=float32)
        >>> axis = jnp.array([2.0, 0.0, 0.0])
        >>> rotation_matrix_along_axis(jnp.pi, axis) @ xyz
        Array([ 7.       ,  0.0000002, -1.       ], dtype=float32)
    """
    axis = jnp.asarray(axis)
    co = jnp.cos(angle)
    si = jnp.sin(angle)
    i = jnp.identity(3, dtype=axis.dtype)
    # Cross product matrix
    x = jnp.array(
        [
            [+0.0, -axis[2], +axis[1]],
            [+axis[2], +0.0, -axis[0]],
            [-axis[1], +axis[0], 0.0],
        ],
    )
    # Outer product matrix
    o = jnp.outer(axis, axis)

    return co * i + si * x + (1 - co) * o


@overload
def fibonacci_lattice(
    n: int,
    dtype: DTypeLike | None = ...,
    *,
    frustum: None = ...,
) -> Float[Array, "{n} 3"]: ...


@overload
def fibonacci_lattice(
    n: int,
    dtype: None = ...,
    *,
    frustum: Float[ArrayLike, "2 2"] | Float[ArrayLike, "2 3"],
) -> Float[Array, "{n} 3"]: ...


def fibonacci_lattice(
    n: int,
    dtype: DTypeLike | None = None,
    *,
    frustum: Float[ArrayLike, "2 2"] | Float[ArrayLike, "2 3"] | None = None,
) -> Float[Array, "{n} 3"]:
    """
    Return a lattice of vertices on the unit sphere.

    This function uses the Fibonacci lattice method :cite:`fibonacci-lattice`
    to generate an almost uniformly distributed set of points on the unit sphere.

    If ``frustum`` is passed, points are distributed in the region defined by the
    frustum's limits.

    Args:
        n: The size of the lattice.
        dtype: The float dtype of the vertices.
        frustum: The spatial region where to sample points.

            The frustum in an array of min. and max. values for
            azimuthal and elevation angles, see :func:`viewing_frustum` for example.

            It is allowed to pass a frustum with distance values, but it will be ignored
            as the distance of from sampled points to origin is always 1.

    Returns:
        Lattice vertices on the unit sphere.

    Raises:
        ValueError: If the provided dtype is not a floating dtype, or if ``n`` is not strictly positive.

    Examples:
        The following example shows how to generate and plot
        a fibonacci lattice.

        .. plotly::

            >>> from differt.geometry import (
            ...     fibonacci_lattice,
            ... )
            >>> from differt.plotting import draw_markers
            >>>
            >>> xyz = fibonacci_lattice(100)
            >>> fig = draw_markers(xyz, marker={"color": xyz[:, 0]}, backend="plotly")
            >>> fig  # doctest: +SKIP
    """
    if n <= 0:
        msg = f"Invalid size {n!r}, must be strictly positive."
        raise ValueError(msg)
    if frustum is not None:
        frustum = jnp.asarray(frustum)
        dtype = frustum.dtype
    elif dtype is not None and not jnp.issubdtype(dtype, jnp.floating):
        msg = f"Unsupported dtype {dtype!r}, must be a floating dtype."
        raise ValueError(msg)

    i = jnp.arange(0.0, n)  # '0.0' forces floating point values

    # Compute the fractional part of (i / phi) for the azimuthal angle.
    #
    # Naively computing ``(i / phi) % 1.0`` in float32 loses precision for
    # large i (e.g. i >= 10_000_000), because the float32 mantissa only has
    # ~7 decimal digits.  At i = 10^7, the product ``i * inv_phi`` is ~6.18e6,
    # and taking ``% 1.0`` amplifies the rounding error so severely that only
    # a handful of unique fractional values remain in the last ~10k points,
    # producing visible "hatching" artifacts in the generated lattice.
    #
    # To fix this we decompose i = q1*m1 + q2*m2 + r  with m1 = 2^18 and
    # m2 = 2^9 (chosen so that every intermediate product stays < 1000),
    # and exploit the identity:
    #
    #     (i * inv_phi) % 1  =  (q1*(inv_phi*m1 % 1)
    #                           + q2*(inv_phi*m2 % 1)
    #                           + r * inv_phi           ) % 1
    #
    # Each term is small enough to preserve full float32 precision.
    inv_phi = 0.6180339887498949  # 1 / phi
    m1 = 262144.0  # 2^18
    m2 = 512.0  # 2^9

    # Pre-compute the fractional parts of inv_phi * m1 and inv_phi * m2,
    # which are constants independent of i.
    inv_phi_m1 = (inv_phi * m1) % 1.0
    inv_phi_m2 = (inv_phi * m2) % 1.0

    # Decompose i into a two-stage quotient-remainder representation.
    q1 = jnp.floor(i / m1)
    rem = i - q1 * m1
    q2 = jnp.floor(rem / m2)
    r = rem - q2 * m2

    # Reconstruct the fractional part with full precision.
    frac = (q1 * inv_phi_m1 + q2 * inv_phi_m2 + r * inv_phi) % 1.0

    if frustum is not None:
        # When a viewing frustum is provided, distribute points uniformly
        # in solid angle within the frustum's polar and azimuthal bounds.
        p_min, a_min = frustum[0, -2:]
        p_max, a_max = frustum[1, -2:]

        # Uniform spacing in cos(polar) ensures equal solid-angle coverage
        # (the Jacobian of the sphere is sin(p) dp da = -d(cos p) da).
        cos_p_min = jnp.cos(p_min)
        cos_p_max = jnp.cos(p_max)
        denom = jnp.where(n > 1, n - 1, 1.0)
        cos_lat = cos_p_min - (cos_p_min - cos_p_max) * (i / denom)
        lat = jnp.arccos(cos_lat)

        # Azimuthal angle uses the quasi-random fractional part to break
        # lattice alignment, mapped into the frustum's azimuthal range.
        a_width = a_max - a_min
        lon = a_min + a_width * frac
        pa = jnp.stack((lat, lon), axis=-1)
    else:
        # Full-sphere Fibonacci lattice: uniform in cos(polar) over [0, pi]
        # and quasi-random in azimuth over [0, 2*pi].
        lat = jnp.arccos(1 - 2 * i / n)
        lon = 2 * jnp.pi * frac
        pa = jnp.stack((lat, lon), axis=-1)

    return spherical_to_cartesian(pa).astype(dtype)


@overload
def assemble_path(
    from_vertex: Float[ArrayLike, "*#batch 3"],
    intermediate_vertices: Float[ArrayLike, "*#batch num_inter_vertices 3"],
    to_vertex: Float[ArrayLike, "*#batch 3"],
) -> Float[Array, "*batch num_inter_vertices+2 3"]: ...


@overload
def assemble_path(
    from_vertex: Float[ArrayLike, "*#batch 3"],
    intermediate_vertices: Float[ArrayLike, "*#batch 3"],
    to_vertex: None = ...,
) -> Float[Array, "*batch 2 3"]: ...


# NOTE: Jaxtyping does not match the correct shape for `intermediate_vertices` and will match
#       `Float[ArrayLike, "*#batch 3"]` instead of `Float[ArrayLike, "*#batch num_inter_vertices 3"]`.
#       This is why we need to disable type checking here. However, the `no_type_check` decorator also
#       suppresses `typing.get_type_hints()`, which is used by Sphinx to extract type annotations for
#       documentation. To work around this, we use `no_type_check` when not generating documentation.
def assemble_path(
    from_vertex: Float[ArrayLike, "*#batch 3"],
    intermediate_vertices: Float[ArrayLike, "*#batch num_inter_vertices 3"]
    | Float[ArrayLike, "*#batch 3"],
    to_vertex: Float[ArrayLike, "*#batch 3"] | None = None,
) -> Float[Array, "*batch num_vertices+2 3"] | Float[Array, "*batch 2 3"]:
    """
    Assemble path vertices by concatenating start-, intermediate, and end-vertices.

    Arrays broadcasting is automatically performed, and the total
    number of vertices per path is simply ``num_inter_vertices+2``.

    Args:
        from_vertex: The starting vertex of the path.
        intermediate_vertices: The intermediate vertices of the path.
            If ``to_vertex`` is not provided, then this argument is interpreted
            as the end vertex of the path.
        to_vertex: The ending vertex of the path.

    Returns:
        Assembled path vertices.
    """
    from_vertex = jnp.asarray(from_vertex)
    intermediate_vertices = jnp.asarray(intermediate_vertices)
    if to_vertex is None:
        to_vertex = intermediate_vertices
        del intermediate_vertices
        batch = jnp.broadcast_shapes(from_vertex.shape[:-1], to_vertex.shape[:-1])
        return jnp.concatenate(
            (
                jnp.broadcast_to(from_vertex[..., None, :], (*batch, 1, 3)),
                jnp.broadcast_to(to_vertex[..., None, :], (*batch, 1, 3)),
            ),
            axis=-2,
        )
    to_vertex = jnp.asarray(to_vertex)
    batch = jnp.broadcast_shapes(
        from_vertex.shape[:-1],
        intermediate_vertices.shape[:-2],
        to_vertex.shape[:-1],
    )

    return jnp.concatenate(
        (
            jnp.broadcast_to(from_vertex[..., None, :], (*batch, 1, 3)),
            jnp.broadcast_to(
                intermediate_vertices, (*batch, *intermediate_vertices.shape[-2:])
            ),
            jnp.broadcast_to(to_vertex[..., None, :], (*batch, 1, 3)),
        ),
        axis=-2,
    )


if not TYPE_CHECKING and not hasattr(typing, "GENERATING_DOCS"):
    assemble_path = no_type_check(assemble_path)


@jax.jit
def min_distance_between_cells(
    cell_vertices: Float[ArrayLike, "*batch 3"],
    cell_ids: Int[ArrayLike, "*batch"],
) -> Float[Array, "*batch"]:
    """
    Compute the minimal (Euclidean) distance between vertices in different cells.

    For every vertex, the minimum distance to another vertex that is not is the same
    cell is computed.

    For an actual application example, see :ref:`multipath_lifetime_map`.

    Args:
        cell_vertices: Vertex coordinates.
        cell_ids: Cell index for each vertex.

    Returns:
        Minimal (Euclidean) distance to a vertex in a different cell.
    """
    cell_vertices = jnp.asarray(cell_vertices)
    cell_ids = jnp.asarray(cell_ids)

    def scan_fun(
        _: None, vertex_and_cell_id: tuple[Float[Array, "3"], Int[Array, ""]]
    ) -> tuple[None, Float[Array, ""]]:
        vertex, cell_id = vertex_and_cell_id
        min_dist = jnp.min(
            jnp.linalg.norm(
                cell_vertices - vertex,
                axis=-1,
            ),
            initial=jnp.inf,
            where=(cell_id != cell_ids),
        )
        return None, min_dist

    return jax.lax.scan(
        scan_fun,
        init=None,
        xs=(
            cell_vertices.reshape(-1, 3),
            cell_ids.reshape(-1),
        ),
    )[1].reshape(cell_ids.shape)


@overload
def viewing_frustum(
    viewing_vertex: Float[ArrayLike, "*#batch 3"],
    world_vertices: Float[ArrayLike, "*#batch num_vertices 3"],
    *,
    active_vertices: Bool[ArrayLike, "*#batch num_vertices"] | None = ...,
    reduce: Literal[False] = False,
) -> Float[Array, "*batch 2 3"]: ...


@overload
def viewing_frustum(
    viewing_vertex: Float[ArrayLike, "*#batch 3"],
    world_vertices: Float[ArrayLike, "*#batch num_vertices 3"],
    *,
    active_vertices: Bool[ArrayLike, "*#batch num_vertices"] | None = ...,
    reduce: Literal[True] = True,
) -> Float[Array, "2 3"]: ...


@eqx.filter_jit
def viewing_frustum(
    viewing_vertex: Float[ArrayLike, "*#batch 3"],
    world_vertices: Float[ArrayLike, "*#batch num_vertices 3"],
    *,
    active_vertices: Bool[ArrayLike, "*#batch num_vertices"] | None = None,
    reduce: bool = False,
) -> Float[Array, "*batch 2 3"] | Float[Array, "2 3"]:
    r"""
    Compute the viewing frustum as seen by one viewer.

    The frustum is a region, expressed in spherical coordinates,
    see :ref:`spherical-coordinates`,
    that fully contains the world vertices.

    Warning:
        The frustum may present wrong results along the polar axis.

        We are still looking at a better way to compute the frustum,
        so feel free to reach out if your have any suggestion.

    Args:
        viewing_vertex: The coordinates of the viewer (i.e., camera).
        world_vertices: World vertex coordinates.
        active_vertices: An optional mask to select which vertices
            to consider when computing the frustum.
        reduce: Whether to reduce batch dimensions.

    Returns:
        The extents (min. and max. values) of the viewing frustum.

    Examples:
        The following example shows how to *launch* rays in a limited
        region of space, to avoid launching rays where no triangles
        would be hit.

        .. plotly::
            :context: reset

            >>> from differt.geometry import (
            ...     fibonacci_lattice,
            ...     viewing_frustum,
            ...     Mesh,
            ... )
            >>> from differt.plotting import draw_rays, reuse, draw_markers
            >>>
            >>> with reuse("plotly") as fig:  # doctest: +SKIP
            ...     tx = jnp.array([0.0, 0.0, 0.0])
            ...     key = jax.random.key(1234)
            ...     draw_markers(tx.reshape(-1, 3), labels=["tx"], showlegend=False)
            ...     for mesh in Mesh.box(with_top=True).translate(tx).iter_objects():
            ...         key, key_color = jax.random.split(key, 2)
            ...         color = r, g, b = jax.random.randint(key_color, (3,), 0, 256)
            ...         center = mesh.bounding_box.mean(axis=0)
            ...         mesh = mesh.translate(5 * (center - tx)).set_face_colors(
            ...             color / 255.0
            ...         )
            ...         mesh.plot(opacity=0.5)
            ...
            ...         frustum = viewing_frustum(
            ...             tx, mesh.triangle_vertices.reshape(-1, 3)
            ...         )
            ...         ray_origins, ray_directions = jnp.broadcast_arrays(
            ...             tx, fibonacci_lattice(20, frustum=frustum)
            ...         )
            ...         ray_origins += 0.5 * ray_directions
            ...         ray_directions *= 2.5  # Scale rays length before plotting
            ...         draw_rays(
            ...             ray_origins,
            ...             ray_directions,
            ...             color=f"rgb({r:f},{g:f},{b:f})",
            ...             showlegend=False,
            ...         )
            >>> fig  # doctest: +SKIP

        This second example shows what happens if you compute the frustum on all the objects
        at the same time, instead of computing one frustum per object (i.e., face).

        .. plotly::
            :context:

            >>> with reuse("plotly") as fig:  # doctest: +SKIP
            ...     tx = jnp.array([0.0, 0.0, 0.0])
            ...     world_vertices = jnp.empty((0, 3))
            ...     draw_markers(tx.reshape(-1, 3), labels=["tx"], showlegend=False)
            ...     for mesh in (
            ...         Mesh
            ...         .box(with_top=True)
            ...         .translate(tx)
            ...         .set_face_colors(jnp.array([1.0, 0.0, 0.0]))
            ...         .iter_objects()
            ...     ):
            ...         center = mesh.bounding_box.mean(axis=0)
            ...         mesh = mesh.translate(5 * (center - tx))
            ...         mesh.plot(opacity=0.5)
            ...
            ...         world_vertices = jnp.concatenate(
            ...             (world_vertices, mesh.triangle_vertices.reshape(-1, 3)),
            ...             axis=0,
            ...         )
            ...
            ...     frustum = viewing_frustum(tx, world_vertices)
            ...     ray_origins, ray_directions = jnp.broadcast_arrays(
            ...         tx, fibonacci_lattice(20 * 6, frustum=frustum)
            ...     )
            ...     ray_origins += 0.5 * ray_directions
            ...     ray_directions *= 2.5  # Scale rays length before plotting
            ...     draw_rays(
            ...         ray_origins,
            ...         ray_directions,
            ...         color="red",
            ...         showlegend=False,
            ...     )
            >>> fig  # doctest: +SKIP

        While the rays cover all the objects, many of them are launching in spatial regions where there
        is not object to hit.

        This third example shows a scenario where TX is far from the mesh,
        where computing the frustum becomes very suitable.

        .. plotly::
            :context:

            >>> with reuse("plotly") as fig:  # doctest: +SKIP
            ...     tx = jnp.array([30.0, 0.0, 20.0])
            ...     draw_markers(tx.reshape(-1, 3), labels=["tx"], showlegend=False)
            ...     mesh = Mesh.box(
            ...         width=10.0, length=20.0, height=3.0, with_top=True
            ...     ).set_face_colors(jnp.array([1.0, 0.0, 0.0]))
            ...     mesh.plot(opacity=0.5)
            ...
            ...     frustum = viewing_frustum(tx, mesh.triangle_vertices.reshape(-1, 3))
            ...     ray_origins, ray_directions = jnp.broadcast_arrays(
            ...         tx, fibonacci_lattice(20 * 6, frustum=frustum)
            ...     )
            ...     ray_origins += 0.5 * ray_directions
            ...     ray_directions *= 40.0  # Scale rays length before plotting
            ...     draw_rays(
            ...         ray_origins,
            ...         ray_directions,
            ...         color="red",
            ...         showlegend=False,
            ...     )
            >>> fig  # doctest: +SKIP

        This fourth example shows how to launch rays in a frustum covering a loaded scene,
        with the viewing vertex (tx) positioned at the center at height z=32.

        .. plotly::
            :context:

            >>> from differt.geometry import Scene, get_sionna_scene
            >>> from differt.plotting import draw_rays, reuse, draw_markers
            >>> from differt.geometry import fibonacci_lattice, viewing_frustum
            >>>
            >>> # Load the simple street canyon scene
            >>> scene_path = get_sionna_scene("simple_street_canyon")
            >>> scene = Scene.load_xml(scene_path)
            >>> tx = jnp.array([0.0, 0.0, 32.0])
            >>>
            >>> with reuse("plotly") as fig:  # doctest: +SKIP
            ...     draw_markers(tx.reshape(-1, 3), labels=["tx"], showlegend=False)
            ...     scene.mesh.plot(opacity=0.5)
            ...
            ...     frustum = viewing_frustum(
            ...         tx, scene.mesh.triangle_vertices.reshape(-1, 3)
            ...     )
            ...     frustum = frustum.at[1, 1].set(jnp.pi)
            ...     ray_origins, ray_directions = jnp.broadcast_arrays(
            ...         tx, fibonacci_lattice(200, frustum=frustum)
            ...     )
            ...     ray_origins += 0.5 * ray_directions
            ...     ray_directions *= 40.0  # Scale rays length before plotting
            ...     draw_rays(
            ...         ray_origins,
            ...         ray_directions,
            ...         color="red",
            ...         showlegend=False,
            ...     )
            >>> fig  # doctest: +SKIP
    """
    world_vertices = jnp.asarray(world_vertices)
    viewing_vertex = jnp.asarray(viewing_vertex)

    # Convert all world vertices to spherical coordinates (r, polar, azimuth)
    # relative to the viewing vertex.
    xyz = world_vertices - viewing_vertex[..., None, :]
    rpa = cartesian_to_spherical(xyz)

    if active_vertices is not None:
        active_vertices: Array = jnp.asarray(active_vertices)

    r, p, a = rpa[..., 0], rpa[..., 1], rpa[..., 2]

    axis = None if reduce else -1

    # ------------------------------------------------------------------
    # Radial and polar bounds: straightforward min/max over active verts.
    # ------------------------------------------------------------------
    r_min = jnp.min(r, axis=axis, where=active_vertices, initial=jnp.inf)
    r_max = jnp.max(r, axis=axis, where=active_vertices, initial=0)
    p_min = jnp.min(p, axis=axis, where=active_vertices, initial=jnp.pi)
    p_max = jnp.max(p, axis=axis, where=active_vertices, initial=0)

    # ------------------------------------------------------------------
    # Azimuthal bounds — handling the -pi / +pi discontinuity.
    #
    # cartesian_to_spherical returns azimuth in [-pi, pi).  When geometry
    # straddles the -pi/+pi boundary (e.g. angles {-170°, +170°}), a
    # naive min/max would give [-170°, +170°] = 340° span, even though
    # the actual span is only 20° across the boundary.
    #
    # Strategy: compute bounds in TWO domains, then keep the tighter one.
    # ------------------------------------------------------------------

    # Domain 1: native [-pi, pi) range.
    a_min = jnp.min(a, axis=axis, where=active_vertices, initial=jnp.pi)
    a_max = jnp.max(a, axis=axis, where=active_vertices, initial=-jnp.pi)

    # Domain 2: shift to [0, 2*pi) — the discontinuity moves to 0/2*pi,
    # which is harmless when geometry is near ±pi.
    two_pi = 2 * jnp.pi
    a_0 = (a + two_pi) % two_pi
    a_0_min = jnp.min(a_0, axis=axis, where=active_vertices, initial=two_pi)
    a_0_max = jnp.max(a_0, axis=axis, where=active_vertices, initial=0)

    # Compare the angular widths in both domains and keep the narrower one,
    # since that is the one that avoids the wrap-around discontinuity.
    a_width = a_max - a_min
    a_0_width = a_0_max - a_0_min

    a_min, a_max = jnp.where(
        a_width > a_0_width,
        jnp.stack((a_0_min, a_0_max)),
        jnp.stack((a_min, a_max)),
    )

    # ------------------------------------------------------------------
    # Full-circle fallback.
    #
    # When the geometry truly surrounds the viewing vertex (e.g. a TX
    # placed inside a corridor that extends in all azimuthal directions),
    # *both* domains produce a large span (> 270°).  Neither domain can
    # represent the coverage compactly, so we fall back to the full
    # circle [-pi, pi] to avoid excluding valid angular regions.
    # ------------------------------------------------------------------
    min_width = jnp.minimum(a_width, a_0_width)
    use_full_circle = min_width > 1.5 * jnp.pi  # > 270°
    a_min = jnp.where(use_full_circle, -jnp.pi, a_min)
    a_max = jnp.where(use_full_circle, jnp.pi, a_max)

    # ------------------------------------------------------------------
    # Polar angle special case.
    #
    # When all vertices share the same polar angle (p_min == p_max),
    # the frustum degenerates to a zero-width band.  We expand it so
    # that rays can still reach the geometry:
    #   - Option A: expand downward to 0 (keep max, widen upward).
    #   - Option B: expand upward to pi (keep min, widen downward).
    # We pick whichever yields the smaller total width.
    # TODO: improve this heuristic for more general degenerate cases.
    # ------------------------------------------------------------------
    p_0_min = p_min
    p_0_max = p_max

    p_min = jnp.where(p_min == p_max, 0.0, p_min)
    p_0_max = jnp.where(p_0_min == p_0_max, jnp.pi, p_0_max)

    p_width = p_max - p_min
    p_0_width = p_0_max - p_0_min

    p_min, p_max = jnp.where(
        p_width > p_0_width,
        jnp.stack((p_0_min, p_0_max)),
        jnp.stack((p_min, p_max)),
    )

    return jnp.stack(
        (
            r_min,
            p_min,
            a_min,
            r_max,
            p_max,
            a_max,
        ),
        axis=-1,
    ).reshape(*r.shape[:-1], 2, 3)


@jax.jit
def cartesian_to_spherical(
    xyz: Float[ArrayLike, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """
    Transform Cartesian coordinates to spherical coordinates.

    See :ref:`conventions` for details.

    Args:
        xyz: Cartesian coordinates.

    Returns:
        Corresponding spherical coordinates.

    .. seealso::

        :func:`spherical_to_cartesian`
    """
    xyz = jnp.asarray(xyz)
    r = jnp.linalg.norm(xyz, axis=-1)
    r: Array = jnp.where(r == 0.0, jnp.ones_like(r), r)
    p = jnp.acos(xyz[..., -1] / r)
    a = jnp.atan2(xyz[..., 1], xyz[..., 0])

    return jnp.stack((r, p, a), axis=-1)


@jax.jit
def spherical_to_cartesian(
    rpa: Float[ArrayLike, "*batch 3"] | Float[ArrayLike, "*batch 2"],
) -> Float[Array, "*batch 3"]:
    """
    Transform spherical coordinates to Cartesian coordinates.

    See :ref:`conventions` for details.

    Args:
        rpa: Spherical coordinates.

            If the radial component is missing, a radius of 1 is assumed.

    Returns:
        Corresponding Cartesian coordinates.

    .. seealso::

        :func:`cartesian_to_spherical`
    """
    rpa = jnp.asarray(rpa)
    p = rpa[..., -2]
    a = rpa[..., -1]

    cp = jnp.cos(p)
    sp = jnp.sin(p)
    ca = jnp.cos(a)
    sa = jnp.sin(a)

    xyz = jnp.stack((sp * ca, sp * sa, cp), axis=-1)

    if rpa.shape[-1] == 3:  # ruff:ignore[magic-value-comparison]
        xyz *= rpa[..., 0, None]

    return xyz


if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'

_T = TypeVar("_T")


class SizedIterator(Iterator[_T], Sized):
    """A custom generic class that is both :class:`Iterator<collections.abc.Iterator>` and :class:`Sized<collections.abc.Sized>`.

    The main purpose of this class is to be able to use
    `tqdm <https://github.com/tqdm/tqdm>`_ utilities
    on iterators and have some meaningful information about how iterations are left.

    Args:
        iter_: The iterator.
        size: The size, i.e., length, of the iterator, or a callable that returns its current length.

    Examples:
        The following example shows how to create a sized iterator.

        >>> from differt.rt import SizedIterator
        >>> l = [1, 2, 3, 4, 5]
        >>> it = SizedIterator(iter=iter(l), size=5)
        >>> len(it)
        5
        >>> it = SizedIterator(iter=iter(l), size=l.__len__)
        >>> len(it)
        5

    """

    __slots__ = ("_iter", "_size")

    def __init__(self, iter: Iterator[_T], size: int | Callable[[], int]) -> None:  # ruff:ignore[builtin-argument-shadowing]
        self._iter = iter
        self._size = size

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> _T:
        return next(self._iter)

    def __len__(self) -> int:
        if isinstance(self._size, int):
            return self._size
        return self._size()


@cache
def generate_all_path_candidates(
    num_primitives: int,
    order: int,
) -> Int[Array, "num_candidates order"]:
    """
    Generate an array of all path candidates for fixed path order and a number of primitives.

    The returned array contains, for each row, an array of
    ``order`` indices indicating the primitive with which the path interacts.

    This list is generated as the list of all paths from one node to
    another, by passing by exactly ``order`` primitives. Calling this function
    is equivalent to calling :func:`itertools.product` with parameters
    ``[0, 1, ..., num_primitives - 1]`` and ``repeat=order``, and removing entries
    containing loops, i.e., two or more consecutive indices that are equal.

    Args:
        num_primitives: The (positive) number of primitives.
        order: The path order. An order less than one returns an empty array.

    Returns:
        An unsigned array with primitive indices on each column. Its number of
        columns is actually equal to
        ``num_primitives * ((num_primitives - 1) ** (order - 1))``.
    """
    return jnp.asarray(
        CompleteGraph(num_primitives).all_paths_array(
            from_=num_primitives,
            to=num_primitives + 1,
            depth=order + 2,
            include_from_and_to=False,
        ),
        dtype=int,
    )


def generate_all_path_candidates_iter(
    num_primitives: int,
    order: int,
) -> SizedIterator[Int[Array, " order"]]:
    """
    Iterator variant of :func:`generate_all_path_candidates`.

    Args:
        num_primitives: The (positive) number of primitives.
        order: The path order.

    Returns:
        An iterator of unsigned arrays with primitive indices.
    """
    it = CompleteGraph(num_primitives).all_paths(
        from_=num_primitives,
        to=num_primitives + 1,
        depth=order + 2,
        include_from_and_to=False,
    )
    m = (jnp.asarray(arr, dtype=int) for arr in it)
    return SizedIterator(m, size=it.__len__)


def generate_all_path_candidates_chunks_iter(
    num_primitives: int,
    order: int,
    chunk_size: int = 1000,
) -> SizedIterator[Int[Array, "chunk_size order"]]:
    """
    Iterator variant of :func:`generate_all_path_candidates`, grouped in chunks of size of max. ``chunk_size``.

    Args:
        num_primitives: The (positive) number of primitives.
        order: The path order.
        chunk_size: The size of each chunk.

    Returns:
        An iterator of unsigned arrays with primitive indices.
    """
    it = CompleteGraph(num_primitives).all_paths_array_chunks(
        from_=num_primitives,
        to=num_primitives + 1,
        depth=order + 2,
        include_from_and_to=False,
        chunk_size=chunk_size,
    )
    m = (jnp.asarray(arr, dtype=int) for arr in it)
    return SizedIterator(m, size=it.__len__)


@overload
def ray_intersect_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: Float[ArrayLike, ""],
) -> tuple[Float[Array, " *batch"], Bool[Array, " *batch"]]: ...


@overload
def ray_intersect_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: None = ...,
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]: ...


@eqx.filter_jit
def ray_intersect_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    *,
    epsilon: Float[ArrayLike, ""] | None = None,
    smoothing_factor: Float[ArrayLike, ""] | None = None,
) -> tuple[Float[Array, " *batch"], Bool[Array, " *batch"] | Float[Array, " *batch"]]:
    """
    Return whether rays intersect corresponding triangles using the Möller-Trumbore algorithm.

    The current implementation closely follows the C++ code from Wikipedia.

    Args:
        ray_origins: Origin vertex.
        ray_directions: Ray direction. The ray end
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: Triangle vertices.
        epsilon: A small tolerance threshold that allows rays
            to hit the triangles slightly outside the actual area.
            A positive value virtually increases the size of the triangles,
            a negative value will have the opposite effect.

            Such a tolerance is especially useful when rays are hitting
            triangle edges, a very common case if geometries are planes
            split into multiple triangles.

            If not specified, the default is ten times the epsilon value
            of the currently used floating point dtype.
        smoothing_factor: If set, hard conditions are replaced with smoothed ones,
            as described in :cite:`fully-eucap2024`, and this argument parameterizes the slope
            of the smoothing function. The second output value is now a real value
            between 0 (:data:`False`) and 1 (:data:`True`).

            For more details, refer to :ref:`smoothing`.

    Returns:
        For each ray, return the scale factor of ``ray_directions`` for the
        vector to reach the corresponding triangle, and whether the intersection
        actually lies inside the triangle.

    Examples:
        The following example shows how to identify triangles that are
        intersected by rays.

        .. plotly::

            >>> import equinox as eqx
            >>> from differt.geometry import fibonacci_lattice
            >>> from differt.plotting import draw_rays
            >>> from differt.rt import (
            ...     ray_intersect_triangle,
            ... )
            >>> from differt.geometry import (
            ...     get_sionna_scene,
            ...     download_sionna_scenes,
            ... )
            >>> from differt.geometry import Scene
            >>>
            >>> download_sionna_scenes()  # doctest: +SKIP
            >>> file = get_sionna_scene("simple_street_canyon")
            >>> scene = Scene.load_xml(file)
            >>> scene = eqx.tree_at(
            ...     lambda s: s.transmitters, scene, jnp.array([-33, 0, 32.0])
            ... )
            >>> ray_origins, ray_directions = jnp.broadcast_arrays(
            ...     scene.transmitters, fibonacci_lattice(25)
            ... )
            >>> # [num_rays=25 num_triangles]
            >>> t, hit = ray_intersect_triangle(
            ...     ray_origins[:, None, :],
            ...     ray_directions[:, None, :],
            ...     scene.mesh.triangle_vertices,
            ... )
            >>> rays_hit = hit.any(axis=1)  # True if rays hit any triangle
            >>> triangles_hit = hit.any(axis=0)  # True if triangles hit by any ray
            >>> ray_directions *= jnp.max(
            ...     t, axis=1, keepdims=True, initial=1.0, where=hit
            ... )  # Scale rays length before plotting
            >>> fig = draw_rays(  # We only plot rays hitting at least one triangle
            ...     ray_origins[rays_hit, :],
            ...     ray_directions[rays_hit, :],
            ...     backend="plotly",
            ...     color="red",
            ...     showlegend=False,
            ... )
            >>> visible_color = jnp.array([0.2, 0.2, 0.2])
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[triangles_hit, :].set(visible_color),
            ... )
            >>> fig = scene.plot(backend="plotly", figure=fig, showlegend=False)
            >>> fig  # doctest: +SKIP
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    triangle_vertices = jnp.asarray(triangle_vertices)

    if epsilon is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        epsilon = 10.0 * jnp.finfo(dtype).eps

    epsilon = jnp.asarray(epsilon)

    # [*batch 3]
    vertex_0 = triangle_vertices[..., 0, :]
    vertex_1 = triangle_vertices[..., 1, :]
    vertex_2 = triangle_vertices[..., 2, :]

    # [*batch 3]
    edge_1 = vertex_1 - vertex_0
    edge_2 = vertex_2 - vertex_0

    # [*batch 3]
    h = jnp.cross(ray_directions, edge_2)

    # [*batch]
    a = jnp.sum(h * edge_1, axis=-1)
    a: Array = jnp.where(a == 0.0, jnp.inf, a)  # Avoid division by zero

    if smoothing_factor is not None:
        hit = smoothing_function(jnp.abs(a) - epsilon, smoothing_factor)
    else:
        hit = jnp.abs(a) > epsilon

    f = 1.0 / a
    s = ray_origins - vertex_0
    u = f * jnp.sum(s * h, axis=-1)

    if smoothing_factor is not None:
        hit = jnp.stack(
            (
                hit,
                smoothing_function(u - 0.0, smoothing_factor),
                smoothing_function(1.0 - u, smoothing_factor),
            ),
            axis=-1,
        ).min(axis=-1, initial=1.0)
    else:
        hit &= (u >= 0.0) & (u <= 1.0)

    q = jnp.cross(s, edge_1)
    v = f * jnp.sum(q * ray_directions, axis=-1)

    if smoothing_factor is not None:
        hit = jnp.stack(
            (
                hit,
                smoothing_function(v - 0.0, smoothing_factor),
                smoothing_function(1.0 - (u + v), smoothing_factor),
            ),
            axis=-1,
        ).min(axis=-1, initial=1.0)
    else:
        hit &= (v >= 0.0) & (u + v <= 1.0)

    t = f * jnp.sum(q * edge_2, axis=-1)

    if smoothing_factor is not None:
        hit = jnp.minimum(hit, smoothing_function(t - epsilon, smoothing_factor))
    else:
        hit &= t > epsilon

    return t, hit


@overload
def ray_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = ...,
    *,
    hit_tol: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: None = ...,
    batch_size: int | None = ...,
    **kwargs: Any,
) -> Bool[Array, " *batch"]: ...


@overload
def ray_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = ...,
    *,
    hit_tol: Float[ArrayLike, ""] | None = ...,
    smoothing_factor: Float[ArrayLike, ""],
    batch_size: int | None = ...,
    **kwargs: Any,
) -> Float[Array, " *batch"]: ...


@eqx.filter_jit
def ray_intersect_any_triangle(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    *,
    hit_tol: Float[ArrayLike, ""] | None = None,
    smoothing_factor: Float[ArrayLike, ""] | None = None,
    batch_size: int | None = 512,
    **kwargs: Any,
) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
    """
    Return whether rays intersect any of the triangles using the Möller-Trumbore algorithm.

    This function should be used when allocating an array of size
    ``*batch num_triangles 3`` (or bigger) is not possible, and you are only interested in
    checking if at least one of the triangles is intersected.

    A triangle is considered to be intersected if
    ``t < (1 - hit_tol) & hit`` evaluates to :data:`True`.

    .. note::

        This function has a faster and more memory-efficient equivalent method:
        :meth:`Mesh.ray_intersect_any_triangle<differt.geometry.Mesh.ray_intersect_any_triangle>`,
        as long as smoothing is not required.

    Args:
        ray_origins: Origin vertex.
        ray_directions: Ray direction. The ray end
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: Triangle vertices.
        active_triangles: Optional active triangle mask.
        hit_tol: The tolerance applied to check if a ray hits another object or not,
            before it reaches the expected position, i.e., the 'interaction' object.

            Using a non-zero tolerance is required as it would otherwise trigger
            false positives.

            If not specified, the default is one hundred times the epsilon value
            of the currently used floating point dtype.
        smoothing_factor: If set, hard conditions are replaced with smoothed ones,
            as described in :cite:`fully-eucap2024`, and this argument parameterizes the slope
            of the smoothing function. The second output value is now a real value
            between 0 (:data:`False`) and 1 (:data:`True`).

            For more details, refer to :ref:`smoothing`.
        batch_size: The number of triangles to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The batch size is automatically adjusted to be the minimum of the number of triangles
            and the specified batch size.

            If :data:`None`, the batch size is set to the number of triangles.
        kwargs: Keyword arguments passed to
            :func:`ray_intersect_triangle`.

    Returns:
        For each ray, whether it intersects with any of the triangles.
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    triangle_vertices = jnp.asarray(triangle_vertices)

    if hit_tol is None:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        hit_tol = 100.0 * jnp.finfo(dtype).eps

    hit_threshold = 1.0 - jnp.asarray(hit_tol)

    num_triangles = triangle_vertices.shape[-3]
    if batch_size is None:
        batch_size = num_triangles
    batch_size = max(min(batch_size, num_triangles), 1)
    num_batches, rem = divmod(num_triangles, batch_size)

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)

    # Combine the batch dimensions
    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )

    if num_triangles == 0:
        # If there are no triangles, there are no intersections
        return (
            jnp.zeros(
                batch,
                dtype=jnp.result_type(ray_origins, ray_directions, triangle_vertices),
            )
            if smoothing_factor is not None
            else jnp.zeros(batch, dtype=bool)
        )

    def map_fn(
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch 3"],
        triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
        active_triangles: Bool[Array, "*#batch num_triangles"] | None = None,
    ) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
        t, hit = ray_intersect_triangle(
            ray_origins[..., None, :],
            ray_directions[..., None, :],
            triangle_vertices,
            smoothing_factor=smoothing_factor,
            **kwargs,
        )
        if smoothing_factor is not None:
            return jnp.minimum(
                hit, smoothing_function(hit_threshold - t, smoothing_factor)
            ).sum(axis=-1, where=active_triangles)
        return ((t < hit_threshold) & hit).any(axis=-1, where=active_triangles)

    def reduce_fn(
        left: Bool[Array, " *batch"] | Float[Array, " *batch"],
        right: Bool[Array, " *batch"] | Float[Array, " *batch"],
    ) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
        if smoothing_factor is not None:
            return (left + right).clip(max=1.0)
        return left | right

    def body_fun(
        batch_index: Int[Array, ""],
        intersect: Bool[Array, " *batch"] | Float[Array, " *batch"],
    ) -> Bool[Array, " *batch"] | Float[Array, " *batch"]:
        start_index = batch_index * batch_size
        batch_of_triangle_vertices = jax.lax.dynamic_slice_in_dim(
            triangle_vertices,
            start_index,
            batch_size,
            axis=-3,
            allow_negative_indices=False,
        )
        batch_of_active_triangles = (
            jax.lax.dynamic_slice_in_dim(
                active_triangles,
                start_index,
                batch_size,
                axis=-1,
                allow_negative_indices=False,
            )
            if active_triangles is not None
            else None
        )
        return reduce_fn(
            intersect,
            map_fn(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                triangle_vertices=batch_of_triangle_vertices,
                active_triangles=batch_of_active_triangles,
            ),
        )

    init_val = (
        jnp.zeros(batch)
        if smoothing_factor is not None
        else jnp.zeros(batch, dtype=jnp.bool)
    )

    intersect = jax.lax.fori_loop(
        0,
        num_batches,
        body_fun,
        init_val=init_val,
    )

    if rem > 0:
        return reduce_fn(
            intersect,
            map_fn(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                triangle_vertices=triangle_vertices[..., -rem:, :, :],
                active_triangles=active_triangles[..., -rem:]
                if active_triangles is not None
                else None,
            ),
        )
    return intersect


@eqx.filter_jit
def triangles_visible_from_vertex(
    vertex: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    num_rays: int = int(1e6),
    batch_size: int | None = 512,
    **kwargs: Any,
) -> Bool[Array, "*batch num_triangles"]:
    """
    Return whether triangles are visible from vertex positions.

    This function uses ray launching and
    :func:`fibonacci_lattice<differt.geometry.fibonacci_lattice>` to estimate
    whether a given triangle can be reached from a specific vertex, i.e., with a ray path,
    without interacting with any other triangle facet.

    It also uses
    :func:`viewing_frustum<differt.geometry.viewing_frustum>` to only
    launch rays in a spatial region that contains triangles.

    .. note::

         This function has a faster and more memory-efficient equivalent method:
         :meth:`Mesh.triangles_visible_from_vertex<differt.geometry.Mesh.triangles_visible_from_vertex>`,
         as long as smoothing is not required.

    Args:
        vertex: Vertex, used as origin of the rays.

            Usually, this would be transmitter position.
        triangle_vertices: Triangle vertices.
        active_triangles: Optional active triangle mask.
        num_rays: The number of rays to launch.

            The larger, the more accurate.
        batch_size: The number of rays to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The batch size is automatically adjusted to be the minimum of the number of rays
            and the specified batch size.

            If :data:`None`, the batch size is set to the number of rays.
        kwargs: Keyword arguments passed to
            :func:`ray_intersect_triangle`.

    Returns:
        Boolean mask, ``True`` if each triangle is visible from the corresponding vertex.

    Examples:
        The following example shows how to identify triangles as
        visible from a given transmitter, coloring them in dark gray.

        .. plotly::
            :context: reset

            >>> import equinox as eqx
            >>> from differt.rt import (
            ...     triangles_visible_from_vertex,
            ... )
            >>> from differt.geometry import (
            ...     Scene,
            ...     get_sionna_scene,
            ...     download_sionna_scenes,
            ... )
            >>>
            >>> download_sionna_scenes()  # doctest: +SKIP
            >>> file = get_sionna_scene("simple_street_canyon")
            >>> scene = Scene.load_xml(file)
            >>> scene = eqx.tree_at(
            ...     lambda s: s.transmitters, scene, jnp.array([-33, 0, 32.0])
            ... )
            >>> visible_triangles = triangles_visible_from_vertex(
            ...     scene.transmitters,
            ...     scene.mesh.triangle_vertices,
            ... )
            >>> visible_color = jnp.array([0.2, 0.2, 0.2])
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[visible_triangles, :].set(visible_color),
            ... )
            >>> fig = scene.plot(backend="plotly")
            >>> fig  # doctest: +SKIP

        In this example, a receiver is placed at the opposite side of the street canyon,
        and its visible triangles are colored in blue. Triangles that are visible from both
        the transmitter and the receiver are colored in yellow.

        .. plotly::
            :context:

            >>> scene = eqx.tree_at(
            ...     lambda s: s.receivers, scene, jnp.array([33, 0, 1.5])
            ... )
            >>> visible_triangles = triangles_visible_from_vertex(
            ...     jnp.stack((scene.transmitters, scene.receivers)),
            ...     scene.mesh.triangle_vertices,
            ... )
            >>> triangles_visible_from_tx = visible_triangles[0, :]
            >>> triangles_visible_from_rx = visible_triangles[1, :]
            >>> visible_by_tx_color = jnp.array([0.2, 0.2, 0.2])
            >>> visible_by_rx_color = jnp.array([0.2, 0.8, 0.2])
            >>> visible_by_both_color = jnp.array([0.8, 0.8, 0.2])
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[triangles_visible_from_tx, :].set(
            ...         visible_by_tx_color
            ...     ),
            ... )
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[triangles_visible_from_rx, :].set(
            ...         visible_by_rx_color
            ...     ),
            ... )
            >>> scene = eqx.tree_at(
            ...     lambda s: s.mesh.face_colors,
            ...     scene,
            ...     scene.mesh.face_colors.at[
            ...         triangles_visible_from_tx & triangles_visible_from_rx, :
            ...     ].set(visible_by_both_color),
            ... )
            >>> fig = scene.plot(backend="plotly")
            >>> fig  # doctest: +SKIP
    """
    vertex = jnp.asarray(vertex)
    triangle_vertices = jnp.asarray(triangle_vertices)
    triangle_centers = triangle_vertices.mean(axis=-2, keepdims=True)
    world_vertices = jnp.concat((triangle_vertices, triangle_centers), axis=-2).reshape(
        *triangle_vertices.shape[:-3], -1, 3
    )

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)
        active_vertices = jnp.repeat(active_triangles, 4, axis=-1)
    else:
        active_vertices = None

    # [*batch 3]
    ray_origins = vertex

    # [*batch 2 3]
    frustum = viewing_frustum(
        ray_origins,
        world_vertices,
        active_vertices=active_vertices,
    )

    batch_size = num_rays if batch_size is None else min(batch_size, num_rays)
    num_batches, rem = divmod(num_rays, batch_size)

    # [*batch num_rays 3]
    ray_directions = jnp.vectorize(
        lambda n, frustum: fibonacci_lattice(n, frustum=frustum),
        excluded={0},
        signature="(2,3)->(n,3)",
    )(num_rays, frustum)

    # Combine the batch dimensions
    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-2],
        ray_directions.shape[:-2],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )

    def update_visible_triangles(
        visible_triangles: Bool[Array, "*#batch num_triangles"],
        visible_indices: Int[Array, "*batch batch_size"],
    ) -> Bool[Array, "*#batch num_triangles"]:
        indices = jnp.indices(visible_triangles.shape, sparse=True)
        indices = (*indices[:-1], visible_indices)
        return visible_triangles.at[indices].set(True, wrap_negative_indices=False)

    def map_fn(
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch batch_size 3"],
        triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
        active_triangles: Bool[Array, "*#batch num_triangles"] | None = None,
    ) -> Int[Array, "*batch batch_size"]:
        indices, _ = first_triangle_hit_by_ray(
            ray_origins[..., None, :],
            ray_directions,
            triangle_vertices[..., None, :, :, :],
            active_triangles=active_triangles[..., None, :]
            if active_triangles is not None
            else None,
            batch_size=None,
            **kwargs,
        )
        return indices

    def body_fun(
        batch_index: Int[Array, ""],
        visible_triangles: Bool[Array, "*batch num_triangles"],
    ) -> Bool[Array, "*batch num_triangles"]:
        start_index = batch_index * batch_size
        batch_of_ray_directions = jax.lax.dynamic_slice_in_dim(
            ray_directions,
            start_index,
            batch_size,
            axis=-2,
            allow_negative_indices=False,
        )
        visible_indices = map_fn(
            ray_origins,
            batch_of_ray_directions,
            triangle_vertices,
            active_triangles,
        )
        return update_visible_triangles(visible_triangles, visible_indices)

    init_val = jnp.zeros((*batch, triangle_vertices.shape[-3]), dtype=jnp.bool)

    visible_triangles = jax.lax.fori_loop(
        0,
        num_batches,
        body_fun,
        init_val=init_val,
    )

    if rem > 0:
        visible_indices = map_fn(
            ray_origins,
            ray_directions[..., -rem:, :],
            triangle_vertices,
            active_triangles,
        )
        return update_visible_triangles(visible_triangles, visible_indices)
    return visible_triangles


@eqx.filter_jit
def first_triangle_hit_by_ray(
    ray_origins: Float[ArrayLike, "*#batch 3"],
    ray_directions: Float[ArrayLike, "*#batch 3"],
    triangle_vertices: Float[ArrayLike, "*#batch num_triangles 3 3"],
    active_triangles: Bool[ArrayLike, "*#batch num_triangles"] | None = None,
    batch_size: int | None = 512,
    **kwargs: Any,
) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
    """
    Return the first triangle hit by each ray.

    This function should be used when allocating an array of size
    ``*batch num_triangles 3`` (or bigger) is not possible, and you are only interested in
    getting the first triangle hit by the ray.

    If two or more triangles are hit at the same distance, the one with the closest center to the ray origin is selected. Two triangles are considered to be hit at the same distance if their distances differ by less than ``100 * eps``, or ten times the ``epsilon`` keyword argument passed to :func:`ray_intersect_triangle`.

    .. note::

        This function has a faster and more memory-efficient equivalent method:
        :meth:`Mesh.first_triangle_hit_by_ray<differt.geometry.Mesh.first_triangle_hit_by_ray>`,
        as long as smoothing is not required.

    Args:
        ray_origins: Origin vertex.
        ray_directions: Ray direction. The ray end
            should be equal to ``ray_origins + ray_directions``.
        triangle_vertices: Triangle vertices.
        active_triangles: Optional active triangle mask.
        batch_size: The number of triangles to process in a single batch.
            This allows to make a trade-off between memory usage and performance.

            The batch size is automatically adjusted to be the minimum of the number of triangles
            and the specified batch size.

            If :data:`None`, the batch size is set to the number of triangles.
        kwargs: Keyword arguments passed to
            :func:`ray_intersect_triangle`.

    Returns:
        Index of and distance to the first triangle hit by each ray.

        If no triangle is hit, the index is set to ``-1`` and
        the distance is set to :data:`inf<numpy.inf>`.
    """
    ray_origins = jnp.asarray(ray_origins)
    ray_directions = jnp.asarray(ray_directions)
    triangle_vertices = jnp.asarray(triangle_vertices)

    if epsilon := kwargs.get("epsilon"):
        epsilon = 10.0 * jnp.asarray(epsilon)
    else:
        dtype = jnp.result_type(ray_origins, ray_directions, triangle_vertices)
        epsilon = jnp.asarray(100.0 * jnp.finfo(dtype).eps)

    num_triangles = triangle_vertices.shape[-3]
    if batch_size is None:
        batch_size = num_triangles
    batch_size = max(min(batch_size, num_triangles), 1)
    num_batches, rem = divmod(num_triangles, batch_size)

    if active_triangles is not None:
        active_triangles = jnp.asarray(active_triangles)

    # Combine the batch dimensions
    batch = jnp.broadcast_shapes(
        ray_origins.shape[:-1],
        ray_directions.shape[:-1],
        triangle_vertices.shape[:-3],
        active_triangles.shape[:-1] if active_triangles is not None else (),
    )

    if num_triangles == 0:
        # If there are no triangles, there are no hits
        return (
            jnp.full(batch, -1, dtype=jnp.int32),
            jnp.full(
                batch,
                jnp.inf,
                dtype=jnp.result_type(ray_origins, ray_directions, triangle_vertices),
            ),
        )

    def reduce_fn(
        left: tuple[Int[Array, " *batch"], Float[Array, " *batch"]],
        right: tuple[Int[Array, " *batch"], Float[Array, " *batch"]],
    ) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
        left_indices, left_t = left
        right_indices, right_t = right
        cond = left_t < right_t
        t = jnp.where(cond, left_t, right_t)
        indices = jnp.where(cond, left_indices, right_indices)
        return indices, t

    def map_fn(
        ray_origins: Float[Array, "*#batch 3"],
        ray_directions: Float[Array, "*#batch 3"],
        triangle_vertices: Float[Array, "*#batch num_triangles 3 3"],
        active_triangles: Bool[Array, "*#batch num_triangles"] | None = None,
    ) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
        t, hit = ray_intersect_triangle(
            ray_origins[..., None, :],
            ray_directions[..., None, :],
            triangle_vertices,
            **kwargs,
        )
        if active_triangles is not None:
            hit &= active_triangles
        t = jnp.where(hit, t, jnp.inf)

        min_idx = jnp.argmin(t, axis=-1)
        min_t = jnp.min(t, axis=-1)

        min_idx = jnp.where(jnp.isinf(min_t), -1, min_idx)
        return min_idx, min_t

    def body_fun(
        batch_index: Int[Array, ""],
        carry: tuple[Int[Array, " *batch"], Float[Array, " *batch"]],
    ) -> tuple[Int[Array, " *batch"], Float[Array, " *batch"]]:
        start_index = batch_index * batch_size
        batch_of_triangle_vertices = jax.lax.dynamic_slice_in_dim(
            triangle_vertices,
            start_index,
            batch_size,
            axis=-3,
            allow_negative_indices=False,
        )
        batch_of_active_triangles = (
            jax.lax.dynamic_slice_in_dim(
                active_triangles,
                start_index,
                batch_size,
                axis=-1,
                allow_negative_indices=False,
            )
            if active_triangles is not None
            else None
        )
        indices, t = map_fn(
            ray_origins,
            ray_directions,
            batch_of_triangle_vertices,
            batch_of_active_triangles,
        )
        return reduce_fn(
            carry,
            (indices + start_index, t),
        )

    init_val = (
        -jnp.ones(batch, dtype=jnp.int32),
        jnp.full(
            batch,
            jnp.inf,
            dtype=jnp.result_type(ray_origins, ray_directions, triangle_vertices),
        ),
    )

    indices, t = jax.lax.fori_loop(
        0,
        num_batches,
        body_fun,
        init_val=init_val,
    )

    if rem > 0:
        rem_indices, rem_t = map_fn(
            ray_origins,
            ray_directions,
            triangle_vertices[..., -rem:, :, :],
            active_triangles[..., -rem:] if active_triangles is not None else None,
        )
        indices, t = reduce_fn(
            (indices, t),
            (
                rem_indices + num_batches * batch_size,
                rem_t,
            ),
        )

    is_finite = jnp.isfinite(t)
    indices = jnp.where(is_finite, indices, -1)
    t = jnp.where(is_finite, t, jnp.inf)
    return (indices, t)

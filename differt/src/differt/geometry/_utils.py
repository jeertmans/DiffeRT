import typing
from typing import TYPE_CHECKING, Literal, no_type_check, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, DTypeLike, Float, Int


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
        vectors: An array of vectors.
        keepdims: If set to :data:`True`, the array of lengths
            will have the same number of dimensions as the input.

    Returns:
        The normalized vector and their length.

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
def perpendicular_vectors(u: Float[ArrayLike, "*batch 3"]) -> Float[Array, "*batch 3"]:
    """
    Generate a vector perpendicular to the input vectors.

    Args:
        u: The array of input vectors.

    Returns:
        An array of vectors perpendicular to the input vectors.

    Examples:
        The following example shows how this function works on basic input vectors.

        >>> from differt.geometry import (
        ...     perpendicular_vectors,
        ... )
        >>>
        >>> u = jnp.array([1.0, 0.0, 0.0])
        >>> perpendicular_vectors(u)
        Array([ 0., -0.,  1.], dtype=float32)
        >>> u = jnp.array([1.0, 1.0, 1.0])
        >>> perpendicular_vectors(u)
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
    Generate ``v`` and ``w``, two other arrays of unit vectors that form with input ``u`` an orthogonal basis.

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
    w = perpendicular_vectors(u)
    v = jnp.cross(w, u)
    v = normalize(v)[0]

    return v, w


@jax.jit(inline=True)
def path_lengths(
    paths: Float[ArrayLike, "*batch path_length 3"],
) -> Float[Array, " *batch"]:
    """
    Compute the path length of each path.

    Each path is exactly made of ``path_length`` vertices.

    Args:
        paths: The array of path vertices.

    Returns:
        The array of path lengths.

    Examples:
        The following example shows how to compute the length of a very simple path.

        >>> from differt.geometry import (
        ...     path_lengths,
        ... )
        >>>
        >>> path = jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        >>> path_lengths(path)
        Array(1., dtype=float32)
        >>> path_lengths(jnp.vstack((path, path[::-1, :])))
        Array(2., dtype=float32)
    """
    paths = jnp.asarray(paths)
    vectors = jnp.diff(paths, axis=-2)
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
        The array of vertices.

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
def assemble_paths(
    from_vertices: Float[ArrayLike, "*#batch 3"],
    intermediate_vertices: Float[ArrayLike, "*#batch num_inter_vertices 3"],
    to_vertices: Float[ArrayLike, "*#batch 3"],
) -> Float[Array, "*batch num_inter_vertices+2 3"]: ...


@overload
def assemble_paths(
    from_vertices: Float[ArrayLike, "*#batch 3"],
    intermediate_vertices: Float[ArrayLike, "*#batch 3"],
    to_vertices: None = ...,
) -> Float[Array, "*batch 2 3"]: ...


# NOTE: Jaxtyping does not match the correct shape for `intermediate_vertices` and will match
#       `Float[ArrayLike, "*#batch 3"]` instead of `Float[ArrayLike, "*#batch num_inter_vertices 3"]`.
#       This is why we need to disable type checking here. However, the `no_type_check` decorator also
#       suppresses `typing.get_type_hints()`, which is used by Sphinx to extract type annotations for
#       documentation. To work around this, we use `no_type_check` when not generating documentation.
def assemble_paths(
    from_vertices: Float[ArrayLike, "*#batch 3"],
    intermediate_vertices: Float[ArrayLike, "*#batch num_inter_vertices 3"]
    | Float[ArrayLike, "*#batch 3"],
    to_vertices: Float[ArrayLike, "*#batch 3"] | None = None,
) -> Float[Array, "*batch num_vertices+2 3"] | Float[Array, "*batch 2 3"]:
    """
    Assemble paths vertices by concatenating start-, intermediate, and end-vertices.

    Arrays broadcasting is automatically performed, and the total
    number of vertices per path is simply ``num_inter_vertices+2``.

    Args:
        from_vertices: The starting vertices of the paths.
        intermediate_vertices: The intermediate vertices of the paths.
            If ``to_vertices`` is not provided, then this argument is interpreted
            as the end vertices of the paths.
        to_vertices: The ending vertices of the paths.

    Returns:
        The assembled path vertices.
    """
    from_vertices = jnp.asarray(from_vertices)
    intermediate_vertices = jnp.asarray(intermediate_vertices)
    if to_vertices is None:
        to_vertices = intermediate_vertices
        del intermediate_vertices
        batch = jnp.broadcast_shapes(from_vertices.shape[:-1], to_vertices.shape[:-1])
        return jnp.concatenate(
            (
                jnp.broadcast_to(from_vertices[..., None, :], (*batch, 1, 3)),
                jnp.broadcast_to(to_vertices[..., None, :], (*batch, 1, 3)),
            ),
            axis=-2,
        )
    to_vertices = jnp.asarray(to_vertices)
    batch = jnp.broadcast_shapes(
        from_vertices.shape[:-1],
        intermediate_vertices.shape[:-2],
        to_vertices.shape[:-1],
    )

    return jnp.concatenate(
        (
            jnp.broadcast_to(from_vertices[..., None, :], (*batch, 1, 3)),
            jnp.broadcast_to(
                intermediate_vertices, (*batch, *intermediate_vertices.shape[-2:])
            ),
            jnp.broadcast_to(to_vertices[..., None, :], (*batch, 1, 3)),
        ),
        axis=-2,
    )


if not TYPE_CHECKING and not hasattr(typing, "GENERATING_DOCS"):
    assemble_paths = no_type_check(assemble_paths)


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
        cell_vertices: The array of vertex coordinates.
        cell_ids: The array of corresponding cell indices.

    Returns:
        The array of minimal distances.
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
        world_vertices: The array of world coordinates.
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
            ...     TriangleMesh,
            ... )
            >>> from differt.plotting import draw_rays, reuse, draw_markers
            >>>
            >>> with reuse("plotly") as fig:
            ...     tx = jnp.array([0.0, 0.0, 0.0])
            ...     key = jax.random.key(1234)
            ...     draw_markers(tx.reshape(-1, 3), labels=["tx"], showlegend=False)
            ...     for mesh in (
            ...         TriangleMesh.box(with_top=True).translate(tx).iter_objects()
            ...     ):
            ...         key, key_color = jax.random.split(key, 2)
            ...         color = r, g, b = jax.random.randint(key_color, (3,), 0, 256)
            ...         center = mesh.bounding_box.mean(axis=0)
            ...         mesh = mesh.translate(5 * (center - tx)).set_face_colors(color)
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
            ...         )  # doctest: +SKIP
            >>> fig  # doctest: +SKIP

        This second example shows what happens if you compute the frustum on all the objects
        at the same time, instead of computing one frustum per object (i.e., face).

        .. plotly::
            :context:

            >>> with reuse("plotly") as fig:
            ...     tx = jnp.array([0.0, 0.0, 0.0])
            ...     world_vertices = jnp.empty((0, 3))
            ...     draw_markers(tx.reshape(-1, 3), labels=["tx"], showlegend=False)
            ...     for mesh in (
            ...         TriangleMesh
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
            ...     )  # doctest: +SKIP
            >>> fig  # doctest: +SKIP

        While the rays cover all the objects, many of them are launching in spatial regions where there
        is not object to hit.

        This third example shows a scenario where TX is far from the mesh,
        where computing the frustum becomes very suitable.

        .. plotly::
            :context:

            >>> with reuse("plotly") as fig:
            ...     tx = jnp.array([30.0, 0.0, 20.0])
            ...     draw_markers(tx.reshape(-1, 3), labels=["tx"], showlegend=False)
            ...     mesh = TriangleMesh.box(
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
            ...     )  # doctest: +SKIP
            >>> fig  # doctest: +SKIP

        This fourth example shows how to launch rays in a frustum covering a loaded scene,
        with the viewing vertex (tx) positioned at the center at height z=32.

        .. plotly::
            :context:

            >>> from differt.scene import TriangleScene, get_sionna_scene
            >>> from differt.plotting import draw_rays, reuse, draw_markers
            >>> from differt.geometry import fibonacci_lattice, viewing_frustum
            >>>
            >>> # Load the simple street canyon scene
            >>> scene_path = get_sionna_scene("simple_street_canyon")
            >>> scene = TriangleScene.load_xml(scene_path)
            >>> tx = jnp.array([0.0, 0.0, 32.0])
            >>>
            >>> with reuse("plotly") as fig:
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
            ...     )  # doctest: +SKIP
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
        xyz: The array of Cartesian coordinates.

    Returns:
        The array of corresponding spherical coordinates.

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
        rpa: The array of spherical coordinates.

            If the radial component is missing, a radius of 1 is assumed.

    Returns:
        The array of corresponding Cartesian coordinates.

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

    if rpa.shape[-1] == 3:  # noqa: PLR2004
        xyz *= rpa[..., 0, None]

    return xyz

from typing import Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, DTypeLike, Float, Int, jaxtyped


@jax.jit
@jaxtyped(typechecker=typechecker)
def pairwise_cross(
    u: Float[Array, "m 3"],
    v: Float[Array, "n 3"],
) -> Float[Array, "m n 3"]:
    """
    Compute the pairwise cross product between two arrays of vectors.

    Args:
        u: First array of vectors.
        v: Second array of vectors.

    Returns:
        A 3D tensor with all cross products.
    """
    return jnp.cross(u[:, None, :], v[None, :, :])


@overload
def normalize(
    vector: Float[Array, "*batch 3"],
    keepdims: Literal[False] = False,
) -> tuple[Float[Array, "*batch 3"], Float[Array, " *batch"]]: ...


@overload
def normalize(
    vector: Float[Array, "*batch 3"],
    keepdims: Literal[True],
) -> tuple[Float[Array, "*batch 3"], Float[Array, " *batch 1"]]: ...


# Workaround currently needed,
# see: https://github.com/microsoft/pyright/issues/9149
@overload
def normalize(
    vector: Float[Array, "*batch 3"],
    keepdims: bool,
) -> (
    tuple[Float[Array, "*batch 3"], Float[Array, " *batch"]]
    | tuple[Float[Array, "*batch 3"], Float[Array, " *batch 1"]]
): ...


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def normalize(
    vector: Float[Array, "*batch 3"],
    keepdims: bool = False,
) -> (
    tuple[Float[Array, "*batch 3"], Float[Array, " *batch"]]
    | tuple[Float[Array, "*batch 3"], Float[Array, " *batch 1"]]
):
    """
    Normalize vectors and also return their length.

    This function avoids division by zero by checking vectors
    with zero-length, and returning unit length instead.

    Args:
        vector: An array of vectors.
        keepdims: If set to :data:`True`, the array of lengths
            will have the same number of dimensions are the input.

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
        (Array([0., 0., 0.], dtype=float32), Array(1., dtype=float32))
    """
    length: Array = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    length = jnp.where(length == 0.0, jnp.ones_like(length), length)

    return vector / length, (length if keepdims else jnp.squeeze(length, axis=-1))


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def orthogonal_basis(
    u: Float[Array, "*batch 3"],
    normalize: bool = True,
) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """
    Generate ``v`` and ``w``, two other arrays of unit vectors that form with input ``u`` an orthogonal basis.

    Args:
        u: The first direction of the orthogonal basis.
            It must have a unit length.
        normalize: Whether the output vectors should be normalized.

            This may be needed, especially for vector ``v``,
            as floating-point error can accumulate so much
            that the vector lengths may diverge from the unit
            length by 10% or even more!

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
        (Array([ 0., -1.,  0.], dtype=float32), Array([ 0.,  0., -1.], dtype=float32))
        >>> u, _ = normalize(jnp.array([1.0, 1.0, 1.0]))
        >>> orthogonal_basis(u)
        (Array([ 0.4082483, -0.8164966,  0.4082483], dtype=float32),
         Array([ 0.7071068,  0.       , -0.7071068], dtype=float32))
    """
    vp = jnp.stack((u[..., 2], -u[..., 0], u[..., 1]), axis=-1)
    w = jnp.cross(u, vp, axis=-1)
    v = jnp.cross(w, u, axis=-1)

    if normalize:
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)

    return v, w


@jax.jit
@jaxtyped(typechecker=typechecker)
def path_lengths(
    paths: Float[Array, "*batch path_length 3"],
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
    vectors = jnp.diff(paths, axis=-2)
    lengths = jnp.linalg.norm(vectors, axis=-1)

    return jnp.sum(lengths, axis=-1)


@jax.jit
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_x_axis(
    angle: Float[ArrayLike, " "],
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
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_y_axis(
    angle: Float[ArrayLike, " "],
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
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_z_axis(
    angle: Float[ArrayLike, " "],
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
@jaxtyped(typechecker=typechecker)
def rotation_matrix_along_axis(
    angle: Float[ArrayLike, " "],
    axis: Float[Array, "3"],
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
    dtype: DTypeLike | None = None,
    *,
    frustum: None = None,
) -> Float[Array, "{n} 3"]: ...


@overload
def fibonacci_lattice(
    n: int,
    dtype: None = None,
    *,
    frustum: Float[Array, "2 2"] | Float[Array, "2 3"],
) -> Float[Array, "{n} 3"]: ...


@jaxtyped(typechecker=typechecker)
def fibonacci_lattice(
    n: int,
    dtype: DTypeLike | None = None,
    *,
    frustum: Float[Array, "2 2"] | Float[Array, "2 3"] | None = None,
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

            Unused if ``frustum`` is passed.
        frustum: The spatial region where to sample points.

            The frustum in an array of min. and max. values for
            azimutal and elevation angles, see :func:`viewing_frustum` for example.

            It is allowed to pass a frustum with distance values, but it will be ignored
            as the distance of from sampled points to origin is always 1.

    Returns:
        The array of vertices.

    Raises:
        ValueError: If the provided dtype is not a floating dtype.

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
    if frustum is not None:
        dtype = frustum.dtype
    elif dtype is not None and not jnp.issubdtype(dtype, jnp.floating):
        msg = f"Unsupported dtype {dtype!r}, must be a floating dtype."
        raise ValueError(msg)

    phi = 1.618033988749895  # golden ratio
    i = jnp.arange(0.0, n)  # '0.0' forces floating point values

    lat = jnp.arccos(1 - 2 * i / n)
    lon = 2 * jnp.pi * i / phi

    pa = jnp.stack((lat, lon), axis=-1)

    if frustum is not None:
        pa %= frustum[1, -2:] - frustum[0, -2:]
        pa += frustum[0, -2:]

    return spherical_to_cartesian(pa).astype(dtype)


@jax.jit
@jaxtyped(typechecker=typechecker)
def assemble_paths(
    *path_segments: Float[Array, "*#batch _num_vertices 3"],
) -> Float[Array, "*#batch path_length 3"]:
    """
    Assemble paths by concatenating path vertices along the second to last axis.

    Arrays broadcasting is automatically performed, and the total
    path length is simply is sum of all the number of vertices.

    Args:
        path_segments: The path segments to assemble together.

            Usually, this will be a 3-tuple of transmitter positions,
            interaction points, and receiver positions.

    Returns:
        The assembled paths.
    """
    batch = jnp.broadcast_shapes(*(arr.shape[:-2] for arr in path_segments))

    return jnp.concatenate(
        tuple(
            jnp.broadcast_to(arr, (*batch, *arr.shape[-2:])) for arr in path_segments
        ),
        axis=-2,
    )


@jax.jit
@jaxtyped(typechecker=typechecker)
def min_distance_between_cells(
    cell_vertices: Float[Array, "*batch 3"],
    cell_ids: Int[Array, "*batch"],
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

    @jaxtyped(typechecker=typechecker)
    def scan_fun(
        _: None, vertex_and_cell_id: tuple[Float[Array, "3"], Int[Array, " "]]
    ) -> tuple[None, Float[Array, " "]]:
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
    viewing_vertex: Float[Array, "*#batch 3"],
    world_vertices: Float[Array, "*#batch num_vertices 3"],
    *,
    optimize: bool = False,
    reduce: Literal[False] = False,
) -> Float[Array, "*batch 2 3"]: ...


@overload
def viewing_frustum(
    viewing_vertex: Float[Array, "*#batch 3"],
    world_vertices: Float[Array, "*#batch num_vertices 3"],
    *,
    optimize: bool = False,
    reduce: Literal[True] = True,
) -> Float[Array, "2 3"]: ...


@overload
def viewing_frustum(
    viewing_vertex: Float[Array, "*#batch 3"],
    world_vertices: Float[Array, "*#batch num_vertices 3"],
    *,
    optimize: bool = False,
    reduce: bool,
) -> Float[Array, "*batch 2 3"] | Float[Array, "2 3"]: ...


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def viewing_frustum(
    viewing_vertex: Float[Array, "*#batch 3"],
    world_vertices: Float[Array, "*#batch num_vertices 3"],
    *,
    reduce: bool = False,
) -> Float[Array, "*batch 2 3"] | Float[Array, "2 3"]:
    r"""
    Compute the viewing frustum as seen by one viewer.

    The frustum is a region, espressed in spherical coordinates,
    see :ref:`spherical-coordinates`,
    that fully contains the world vertices.

    Warning:
        The frustum may present wrong results along the polar axis.

        We are still looking at a better way to compute the frustum,
        so feel free to reach out if your have any suggestion.

    Args:
        viewing_vertex: The coordinates of the viewer (i.e., camera).
        world_vertices: The array of world coordinates.
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
            ...         mesh.plot()
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
            ...             line={"color": f"rgb({float(r)},{float(g)},{float(b)})"},
            ...             mode="lines",
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
            ...         TriangleMesh.box(with_top=True)
            ...         .translate(tx)
            ...         .set_face_colors(jnp.array([1.0, 0.0, 0.0]))
            ...         .iter_objects()
            ...     ):
            ...         center = mesh.bounding_box.mean(axis=0)
            ...         mesh = mesh.translate(5 * (center - tx))
            ...         mesh.plot()
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
            ...         line={"color": "red"},
            ...         mode="lines",
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
            ...     mesh.plot()
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
            ...         line={"color": "red"},
            ...         mode="lines",
            ...         showlegend=False,
            ...     )  # doctest: +SKIP
            >>> fig  # doctest: +SKIP
    """
    xyz = world_vertices - viewing_vertex[..., None, :]
    rpa = cartesian_to_spherical(xyz)

    r, p, a = rpa[..., 0], rpa[..., 1], rpa[..., 2]

    if reduce:
        r = r.ravel()
        p = p.ravel()
        a = a.ravel()

    r_min = jnp.min(r, axis=-1)
    r_max = jnp.max(r, axis=-1)
    p_min = jnp.min(p, axis=-1)
    p_max = jnp.max(p, axis=-1)
    a_min = jnp.min(a, axis=-1)
    a_max = jnp.max(a, axis=-1)

    # The discontinuity for azimutal angles near -pi;pi can create
    # issues, leading to a larger angular sector that expected.

    # We map azimutal angles from [-pi;pi[ to [0;2pi[.
    two_pi = 2 * jnp.pi

    a_0 = (a + two_pi) % two_pi
    a_0_min = jnp.min(a_0, axis=-1)
    a_0_max = jnp.max(a_0, axis=-1)

    a_width = a_max - a_min
    a_0_width = a_0_max - a_0_min

    a_min, a_max = jnp.where(
        a_width > a_0_width,
        jnp.stack((a_0_min, a_0_max)),
        jnp.stack((a_min, a_max)),
    )

    # For polar angle, we 'try' to fix a similar issue.
    # TODO: improve this.
    p_0_min = p_max
    p_0_max = p_min

    p_min = jnp.where(p_min == p_max, 0.0, p_min)
    p_0_max = jnp.where(p_0_min == p_0_max, jnp.pi, p_0_max)

    p_width = p_max - p_min
    p_0_width = p_0_max - p_0_min

    p_min, p_max = jnp.where(
        p_width > p_0_width,
        jnp.stack((p_0_min, p_0_max)),
        jnp.stack((p_min, p_max)),
    )

    return jnp.stack((
        r_min,
        p_min,
        a_min,
        r_max,
        p_max,
        a_max,
    )).reshape(*r.shape[:-1], 2, 3)


@jax.jit
@jaxtyped(typechecker=typechecker)
def cartesian_to_spherical(xyz: Float[Array, "*batch 3"]) -> Float[Array, "*batch 3"]:
    """
    Transform cartesian coordinates to spherical coordinates.

    See :ref:`conventions` for details.

    Args:
        xyz: The array of cartesian coordinates.

    Returns:
        The array of corresponding spherical coordinates.

    .. seealso::

        :func:`spherical_to_cartesian`
    """
    r = jnp.linalg.norm(xyz, axis=-1)
    p = jnp.arccos(xyz[..., -1] / r)
    a = jnp.arctan2(xyz[..., 1], xyz[..., 0])

    return jnp.stack((r, p, a), axis=-1)


@jax.jit
@jaxtyped(typechecker=typechecker)
def spherical_to_cartesian(
    rpa: Float[Array, "*batch 3"] | Float[Array, "*batch 2"],
) -> Float[Array, "*batch 3"]:
    """
    Transform spherical coordinates to cartisian coordinates.

    See :ref:`conventions` for details.

    Args:
        rpa: The array of spherical coordinates.

            If the radial component is missing, a radius of 1 is assumed.

    Returns:
        The array of corresponding cartesian coordinates.

    .. seealso::

        :func:`cartesian_to_spherical`
    """
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

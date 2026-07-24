from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int

from differt.geometry import normalize, path_length, perpendicular_vector

from ._constants import c
from ._interaction_type import InteractionType


@jax.jit
def length_to_delay(
    length: Float[ArrayLike, " *#batch"],
    speed: Float[ArrayLike, " *#batch"] = c,
) -> Float[Array, " *batch"]:
    """
    Compute the delay, in seconds, corresponding to the length.

    Args:
        length: Input length (in meters).
        speed: Speed (in m/s).
            Defaults to the speed of light in vacuum.

    Returns:
        The path delay.

    Examples:
        The following example shows how to compute a delay from a length.

        >>> from differt.em import c
        >>> from differt.em import (
        ...     length_to_delay,
        ... )
        >>>
        >>> length = jnp.array([1.0, 2.0, 4.0])
        >>> length_to_delay(length) * c
        Array([1., 2., 4.], dtype=float32)
        >>> length_to_delay(length, speed=2.0)
        Array([0.5, 1. , 2. ], dtype=float32)
    """
    return jnp.asarray(length) / jnp.asarray(speed)


@jax.jit
def path_delay(
    path: Float[ArrayLike, "*batch path_length 3"],
    **kwargs: Any,
) -> Float[Array, " *batch"]:
    """
    Compute the path delay, in seconds, of the path.

    The path is exactly made of ``path_length`` vertices.

    Args:
        path: Input path.
        kwargs: Keyword arguments passed to
            :func:`length_to_delay`.

    Returns:
        The path delay.

    Examples:
        The following example shows how to compute the delay of a very simple path.

        >>> from differt.em import c
        >>> from differt.em import (
        ...     path_delay,
        ... )
        >>>
        >>> path = jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        >>> path_delay(path) * c
        Array(1., dtype=float32)
        >>> path_delay(path, speed=2.0)
        Array(0.5, dtype=float32)
    """
    length = path_length(path)

    return length_to_delay(length, **kwargs)


@jax.jit
def sp_directions(
    k_i: Float[ArrayLike, "*#batch 3"],
    k_r: Float[ArrayLike, "*#batch 3"],
    normals: Float[ArrayLike, "*#batch 3"],
) -> tuple[
    tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]],
    tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]],
]:
    """
    Compute the directions of the local s and p components, before and after reflection, relative to the propagation direction and a local normal.

    Args:
        k_i: The propagation direction of the incident field.

            The vector must have a unit length.
        k_r: The propagation direction of the reflected field.

            The vector must have a unit length.
        normals: The local normal.

            The vector must have a unit length.

    Returns:
        The s and p directions, before and after reflection.

    Examples:
        The following example shows how to compute and display the direction of
        the s and p components before and after reflection on a spherical surface.

        .. plotly::

            >>> import plotly.graph_objects as go
            >>> from differt.geometry import normalize, spherical_to_cartesian
            >>> from differt.em import (
            ...     sp_directions,
            ... )

            We generate a grid of points on a spherical surface.

            >>> u, v = jnp.meshgrid(
            ...     jnp.linspace(3 * jnp.pi / 8, 5 * jnp.pi / 8, 100),
            ...     jnp.linspace(-jnp.pi / 8, jnp.pi / 8),
            ... )
            >>> pa = jnp.stack((u, v), axis=-1)
            >>> xyz = spherical_to_cartesian(pa)
            >>> fig = go.Figure()
            >>> fig.add_trace(
            ...     go.Surface(
            ...         x=xyz[..., 0],
            ...         y=xyz[..., 1],
            ...         z=xyz[..., 2],
            ...         colorscale=["blue", "blue"],
            ...         opacity=0.5,
            ...         showscale=False,
            ...     )
            ... )  # doctest: +SKIP

            Plotly does not provide a nice function to draw 3D vectors, so we create one.

            >>> def add_vector(fig, orig, dest, color="red", name=None, dashed=False):
            ...     dir = dest - orig
            ...     end = orig + 0.9 * dir
            ...     fig.add_trace(
            ...         go.Scatter3d(
            ...             x=[orig[0], end[0]],
            ...             y=[orig[1], end[1]],
            ...             z=[orig[2], end[2]],
            ...             mode="lines",
            ...             line_color=color,
            ...             showlegend=False,
            ...             line_dash="dashdot" if dashed else None,
            ...             legendgroup=name,
            ...         )
            ...     )
            ...     dir = 0.1 * dir
            ...     fig.add_trace(
            ...         go.Cone(
            ...             x=[end[0]],
            ...             y=[end[1]],
            ...             z=[end[2]],
            ...             u=[dir[0]],
            ...             v=[dir[1]],
            ...             w=[dir[2]],
            ...             colorscale=[color, color],
            ...             sizemode="raw",
            ...             showscale=False,
            ...             showlegend=True,
            ...             name=name,
            ...             hoverinfo="name",
            ...             opacity=0.5 if dashed else None,
            ...             legendgroup=name,
            ...         )
            ...     )

            We then place TX and RX points, and determine direction vectors,
            as well as direction of local s and p components.

            >>> reflection_point = jnp.array([1.0, 0.0, 0.0])
            >>> angle = jnp.pi / 4
            >>> cos = jnp.cos(angle)
            >>> sin = jnp.sin(angle)
            >>> normal = jnp.array([1.0, 0.0, 0.0])
            >>> tx = reflection_point + jnp.array([cos, 0.0, +sin])
            >>> rx = reflection_point + jnp.array([cos, 0.0, -sin])
            >>> k_i = reflection_point - tx
            >>> k_r = rx - reflection_point
            >>> (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, normal)
            >>> # ... (lines omitted for brevity)

            Finally, we draw all the vectors and markers.

            >>> fig.add_trace(
            ...     go.Scatter3d(
            ...         x=[tx[0], rx[0]],
            ...         y=[tx[1], rx[1]],
            ...         z=[tx[2], rx[2]],
            ...         mode="markers+text",
            ...         text=["TX", "RX"],
            ...         marker_color="black",
            ...         showlegend=False,
            ...     )
            ... )  # doctest: +SKIP
            >>> add_vector(fig, tx, reflection_point, color="magenta", name="incident")
            >>> add_vector(fig, reflection_point, rx, color="magenta", name="reflected")
            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + normal,
            ...     color="blue",
            ...     name="normal",
            ... )
            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + 0.5 * e_i_s,
            ...     color="orange",
            ...     name="s-component (incident)",
            ... )
            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + 0.5 * e_i_p,
            ...     color="orange",
            ...     name="p-component (incident)",
            ... )

            We do the same, but for the reflected field.

            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + 0.5 * e_r_s,
            ...     color="green",
            ...     name="s-component (reflected)",
            ...     dashed=True,
            ... )
            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + 0.5 * e_r_p,
            ...     color="green",
            ...     name="p-component (reflected)",
            ... )
            >>> fig  # doctest: +SKIP
    """
    k_i = jnp.asarray(k_i)
    k_r = jnp.asarray(k_r)
    normals = jnp.asarray(normals)
    e_i_s, e_i_s_norm = normalize(jnp.cross(k_i, normals), keepdims=True)
    # Alternative vectors if normal is parallel to k_i
    normal_incidence = e_i_s_norm == 0.0
    e_i_s: Array = jnp.where(
        normal_incidence,
        perpendicular_vector(k_i),
        e_i_s,
    )
    e_i_p = normalize(jnp.cross(e_i_s, k_i))[0]
    e_r_s = e_i_s
    e_r_p = normalize(jnp.cross(e_r_s, k_r))[0]

    return (e_i_s, e_i_p), (e_r_s, e_r_p)


@jax.jit
def sp_rotation_matrix(
    e_a_s: Float[ArrayLike, "*#batch 3"],
    e_a_p: Float[ArrayLike, "*#batch 3"],
    e_b_s: Float[ArrayLike, "*#batch 3"],
    e_b_p: Float[ArrayLike, "*#batch 3"],
) -> Float[Array, "*batch 2 2"]:
    """
    Return the rotation matrix to convert the s and p components from one base to another.

    All input vectors must have a unit length, and the direction of propagation must be the same.
    The latter is equivalent to ensuring that all four vectors are coplanar.

    Args:
        e_a_s: s-component direction of the first field basis.
        e_a_p: p-component direction of the first field basis.
        e_b_s: s-component direction of the second field basis.
        e_b_p: p-component direction of the second field basis.

    Returns:
        Rotation matrix from basis ``(e_a_s, e_a_p)`` to ``(e_b_s, e_b_p)``.
    """
    e_a_s = jnp.asarray(e_a_s)
    e_a_p = jnp.asarray(e_a_p)
    e_b_s = jnp.asarray(e_b_s)
    e_b_p = jnp.asarray(e_b_p)
    r11 = jnp.sum(e_b_s * e_a_s, axis=-1, keepdims=True)
    r12 = jnp.sum(e_b_s * e_a_p, axis=-1, keepdims=True)
    r21 = jnp.sum(e_b_p * e_a_s, axis=-1, keepdims=True)
    r22 = jnp.sum(e_b_p * e_a_p, axis=-1, keepdims=True)

    r11, r12, r21, r22 = jnp.broadcast_arrays(r11, r12, r21, r22)

    batch = r11.shape[:-1]

    return jnp.concatenate((r11, r12, r21, r22), axis=-1).reshape(*batch, 2, 2)


@jax.jit
def transition_matrix(
    vertices: Float[ArrayLike, "*batch path_length 3"],
    objects: Float[ArrayLike, "*batch path_length"],
    interaction_types: Int[ArrayLike, "*batch path_length-2"],
    object_normals: Float[ArrayLike, "*batch path_length 3"],
) -> Float[Array, "*batch 2 2"]:
    """
    Compute the transition matrix, ...

    .. warning::

        Do not use this function yet, it is not implemented!
    """
    vertices = jnp.asarray(vertices)
    objects = jnp.asarray(objects)
    interaction_types = jnp.asarray(interaction_types)
    object_normals = jnp.asarray(object_normals)

    if any(x.dtype == jnp.float64 for x in (vertices, object_normals)):
        cdtype = jnp.complex128
    else:
        cdtype = jnp.complex64

    # [*batch 2 2]
    mat = jnp.zeros((vertices.shape[:-2], 2, 2), dtype=cdtype)

    v = jnp.diff(vertices, axis=-2)
    k, s = normalize(v)
    _k_i, _s_i = k[..., :-1, :], s[..., :-1, :]
    _k_r, _s_r = k[..., +1:, :], s[..., +1:, :]

    mat_r = mat  # TODO: fixme

    mat = jnp.where(interaction_types == InteractionType.REFLECTION, mat_r, mat)

    raise NotImplementedError


@jax.jit(static_argnames=("dB",))
def fspl(
    d: Float[ArrayLike, " *#batch"],
    f: Float[ArrayLike, " *#batch"],
    *,
    dB: bool = False,  # ruff:ignore[invalid-argument-name]
) -> Float[Array, " *batch"]:
    """
    Compute the free-space path loss (FSPL), optionally in dB.

    See :cite:`fspl` for more information.

    Args:
        d: Distance (in meters).
        f: Frequency (in Hertz).
        dB: Whether to return the result in dB.

    Returns:
        Free-space path loss.
    """
    if dB:
        return 20 * jnp.log10(d) + 20 * jnp.log10(f) - 147.55221677811662

    return jax.lax.integer_pow(4 * jnp.pi * d * f / c, 2)

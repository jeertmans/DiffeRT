from typing import Any

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, jaxtyped

from differt.geometry import normalize, path_lengths, perpendicular_vectors

from ._constants import c


@jax.jit
@jaxtyped(typechecker=typechecker)
def lengths_to_delays(
    lengths: Float[Array, " *#batch"],
    speed: Float[ArrayLike, " *#batch"] = c,
) -> Float[Array, " *#batch"]:
    """
    Compute the delay, in seconds, corresponding to each length.

    Args:
        lengths: The array of lengths (in meters).
        speed: The speed (in meters per second)
            used to compute the delay. This can be
            an array of speeds. Default is the speed
            of light in vacuum.

    Returns:
        The array of path delays.

    Examples:
        The following example shows how to compute a delay from a length.

        >>> from differt.em import c
        >>> from differt.em import (
        ...     lengths_to_delays,
        ... )
        >>>
        >>> lengths = jnp.array([1.0, 2.0, 4.0])
        >>> lengths_to_delays(lengths) * c
        Array([1., 2., 4.], dtype=float32)
        >>> lengths_to_delays(lengths, speed=2.0)
        Array([0.5, 1. , 2. ], dtype=float32)
    """
    return lengths / speed


@jax.jit
@jaxtyped(typechecker=typechecker)
def path_delays(
    paths: Float[Array, "*batch path_length 3"],
    **kwargs: Any,
) -> Float[Array, " *batch"]:
    """
    Compute the path delay, in seconds, of each path.

    Each path is exactly made of ``path_length`` vertices.

    Args:
        paths: The array of path vertices.
        kwargs: Keyword arguments passed to
            :func:`lengths_to_delays`.

    Returns:
        The array of path delays.

    Examples:
        The following example shows how to compute the delay of a very simple path.

        >>> from differt.em import c
        >>> from differt.em import (
        ...     path_delays,
        ... )
        >>>
        >>> path = jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        >>> path_delays(path) * c
        Array(1., dtype=float32)
        >>> path_delays(path, speed=2.0)
        Array(0.5, dtype=float32)
    """
    lengths = path_lengths(paths)

    return lengths_to_delays(lengths, **kwargs)


@jax.jit
@jaxtyped(typechecker=typechecker)
def sp_directions(
    k_i: Float[Array, "*#batch 3"],
    k_r: Float[Array, "*#batch 3"],
    normals: Float[Array, "*#batch 3"],
) -> tuple[
    tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]],
    tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]],
]:
    """
    Compute the directions of the local s and p components, before and after reflection, relative to the propagation direction and a local normal.

    Args:
        k_i: The array of propagation direction of incident fields.

            Each vector must have a unit length.
        k_r: The array of propagation direction of reflected fields.

            Each vector must have a unit length.
        normals: The array of local normals.

            Each vector must have a unit length.

    Returns:
        The array of s and p directions, before and after reflection.

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
    e_i_s, e_i_s_norm = normalize(jnp.cross(k_i, normals), keepdims=True)
    # Alternative vectors if normal is parallel to k_i
    normal_incidence = e_i_s_norm == 0.0
    e_i_s: Array = jnp.where(
        normal_incidence,
        perpendicular_vectors(k_i),
        e_i_s,
    )  # type: ignore[reportTypeAssignment]
    e_i_p = normalize(jnp.cross(e_i_s, k_i))[0]
    e_r_s = e_i_s
    e_r_p = normalize(jnp.cross(e_r_s, k_r))[0]

    return (e_i_s, e_i_p), (e_r_s, e_r_p)

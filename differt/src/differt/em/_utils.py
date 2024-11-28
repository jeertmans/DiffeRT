from typing import Any

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, jaxtyped

from differt.geometry import normalize, path_lengths

from .constants import c


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
    propagation_directions: Float[Array, "*#batch 3"],
    normals: Float[Array, "*#batch 3"],
) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """
    Compute the directions of the local s and p components, relative to the propagation direction and a local normal or a local edge direction..

    Args:
        propagation_directions: The array of propagation directions.

            Each vector must have a unit length.
        normals: The array of local normals or local edge directions.

            Each vector must have a unit length.

    Returns:
        The array of s and p directions.

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
            >>> incident_dir = reflection_point - tx
            >>> reflection_dir = rx - reflection_point
            >>> e_s, e_p = sp_directions(incident_dir, normal)

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
            ...     reflection_point + 0.5 * e_s,
            ...     color="orange",
            ...     name="s-component (incident)",
            ... )
            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + 0.5 * e_p,
            ...     color="orange",
            ...     name="p-component (incident)",
            ... )

            We do the same, but for the reflected field.

            >>> e_s, e_p = sp_directions(reflection_dir, normal)
            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + 0.5 * e_s,
            ...     color="green",
            ...     name="s-component (reflected)",
            ...     dashed=True,
            ... )
            >>> add_vector(
            ...     fig,
            ...     reflection_point,
            ...     reflection_point + 0.5 * e_p,
            ...     color="green",
            ...     name="p-component (reflected)",
            ... )
            >>> fig  # doctest: +SKIP
    """
    e_s = normalize(jnp.cross(propagation_directions, normals))[0]
    e_p = jnp.cross(propagation_directions, e_s)

    return e_s, e_p

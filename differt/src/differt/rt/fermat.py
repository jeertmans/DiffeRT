"""
Path tracing utilities that utilize Fermat's principle.

Fermat's principle states that the path taken by a ray between two
given points is the path that can be traveled in the least time
:cite:`fermat-principle`. In a homogeneous medium,
this means that the path of least time is also the path of last distance.

As a result, this module offers minimization methods for finding ray paths.
"""

from functools import partial
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from differt.geometry.utils import orthogonal_basis
from differt.utils import minimize


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def fermat_path_on_planar_mirrors(
    from_vertices: Float[Array, "*batch 3"],
    to_vertices: Float[Array, "*batch 3"],
    mirror_vertices: Float[Array, "num_mirrors 3"],
    mirror_normals: Float[Array, "num_mirrors 3"],
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
            :py:func:`minimize<differt.utils.minimize>`.

    Returns:
        An array of ray paths obtained using Fermat's principle.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`jax.numpy.concatenate`:

        .. code-block:: python

            paths = fermat_path_on_planar_mirrors(
                from_vertices,
                to_vertices,
                mirror_vertices,
                mirror_normals,
            )

            full_paths = jnp.concatenate(
                (from_vertices[..., None, :], got, to_vertices[..., None, :]),
                axis=-2,
            )
    """
    batch = from_vertices.shape[:-1]
    num_mirrors = mirror_vertices.shape[0]
    num_unknowns = 2 * num_mirrors

    mirror_directions_1, mirror_directions_2 = orthogonal_basis(mirror_normals)

    @jaxtyped(typechecker=typechecker)
    def st_to_xyz(
        st: Float[Array, "*batch num_unknowns"],
    ) -> Float[Array, "*batch num_mirrors 3"]:
        s = st[..., 0::2, None]
        t = st[..., 1::2, None]
        return mirror_vertices + mirror_directions_1 * s + mirror_directions_2 * t

    @jaxtyped(typechecker=typechecker)
    def loss(  # noqa: PLR0917
        st: Float[Array, "*batch num_unknowns"],
        from_: Float[Array, "*batch 3"],
        to: Float[Array, "*batch 3"],
    ) -> Float[Array, " *batch"]:
        xyz = st_to_xyz(st)
        paths = jnp.concatenate(
            (from_[..., None, :], xyz, to[..., None, :]),
            axis=-2,
        )
        vectors = jnp.diff(paths, axis=-2)
        lengths = jnp.linalg.norm(vectors, axis=-1)
        return jnp.sum(lengths, axis=-1)

    st0 = jnp.zeros((*batch, num_unknowns))
    st, _ = minimize(
        loss,
        st0,
        args=(from_vertices, to_vertices),
        **kwargs,
    )

    return st_to_xyz(st)

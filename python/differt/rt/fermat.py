"""
Path tracing utilities that utilize Fermat's principle.

Fermat's principle states that the path taken by a ray between two
given points is the path that can be traveled in the least time
:cite:`fermat-principle`. In a homogeneous medium,
this means that the path of least time is also the path of last distance.

As a result, this module offers minimization methods for finding ray paths.
"""
import jax.numpy as jnp

from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

from ..utils import minimize


@jaxtyped(typechecker=typechecker)
def fermat_path_on_planar_surfaces(
    from_vertices: Float[Array, "*batch 3"],
    to_vertices: Float[Array, "*batch 3"],
    mirror_vertices: Float[Array, "num_mirrors *batch 3"],
    mirror_normals: Float[Array, "num_mirrors *batch 3"],
) -> Float[Array, "num_mirrors *batch 3"]:
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
            length and if perpendicular to the corresponding mirror.

    Return:
        An array of ray paths obtained using Fermat's principle.

    .. note::

        The paths do not contain the starting and ending vertices.

        You can easily create the complete ray paths using
        :func:`jax.numpy.concatenate`:

        .. code-block:: python

            paths = fermat_path_on_planar_surfaces(
                from_vertices,
                to_vertices,
                mirror_vertices,
                mirror_normals,
            )

            full_paths = jnp.concatenate(
                (
                    from_vertices[
                        None,
                        ...,
                    ],
                    paths,
                    to_vertices[
                        None,
                        ...,
                    ],
                )
            )
    """
    num_mirrors, *batch, _ = mirror_vertices.shape
    num_unknowns = 2 * num_mirrors
    mirror_directions_1 = ...
    mirror_directions_2 = ...

    def parametric_to_cartesian(t, origins, directions_1, directions_2):
        return origins + directions_1 * t[..., 0::2] + directions_2 * t[..., 1::2]

    def loss(st, o, d1, d2):
        s = st[..., 0::2]
        t = st[..., 1::2]
        xyz = o + d1 * s + d2 * t
        vectors = jnp.diff(xyz, axis=-1)
        lengths = jnp.linalg.norm(vectors, axis=-1)
        return jnp.sum(lengths, axis=-1)

    st0 = jnp.zeros((*batch, num_unknowns))
    st, _ = minimize(loss,  st0, fun_args=(mirror_vertices, mirror_directions_1, mirror_directions_2))

    return st

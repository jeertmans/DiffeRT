import jax.numpy as jnp
from jaxtyping import Array, Float

from differt.geometry import normalize
from differt.scene import TriangleScene


def basis_for_canonical_frame(
    tx: Float[Array, "3"],
    rx: Float[Array, "3"],
) -> tuple[Float[Array, "3 3"], Float[Array, " "]]:
    """
    Compute the basis for the canonical frame where the z-axis is aligned with the projection of the tx-rx direction on the xy-plane.

    Args:
        tx: Transmitter position.
        rx: Receiver position.

    Returns:
        A tuple containing:
            - A 3x3 array representing the rotation matrix to the canonical frame.
            - A scalar representing the scale (distance between tx and rx).
    """
    w, scale = normalize(rx - tx)
    ref_axis = jnp.array([0.0, 0.0, 1.0])
    u, _ = normalize(jnp.cross(w, ref_axis))
    v, _ = normalize(jnp.cross(w, u))
    return jnp.stack((u, v, w)), scale


def geometric_transformation(
    xyz: Float[Array, "num_objects num_vertices_per_object 3"],
    tx: Float[Array, "3"],
    rx: Float[Array, "3"],
) -> Float[Array, "num_objects num_vertices_per_object 3"]:
    """
    Apply the geometric transformation to the vertices of the triangles such that the output is invariant to:
    - Translation
    - Rotation (along the z-axis)
    - Scale

    Args:
        xyz: Array of object vertices.
        tx: Transmitter position.
        rx: Receiver position.

    Returns:
        Array of transformed triangle vertices.
    """
    basis, scale = basis_for_canonical_frame(tx, rx)
    xyz = xyz - tx
    xyz = xyz / scale
    xyz = xyz @ basis.T
    return xyz


def unpack_scene(
    scene: TriangleScene,
) -> tuple[
    Float[Array, "num_triangles 3 3"], Float[Array, "3"], Float[Array, "3"]
]:
    """
    Unpack the scene into its components.

    Args:
        scene: The scene to unpack.

    Returns:
        A tuple containing:
            - The triangle vertices.
            - The transmitter position.
            - The receiver position.
    """
    return (
        scene.mesh.triangle_vertices,
        scene.transmitters.reshape(3),
        scene.receivers.reshape(3),
    )

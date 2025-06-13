"""Utilities to work with the DeepMIMO framework."""

__all__ = ("ArrayType", "DeepMIMO", "export")

from collections.abc import Iterable, Mapping
from dataclasses import KW_ONLY, asdict
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from differt.em import (
    InteractionType,
    Material,
    c,
    materials,
    reflection_coefficients,
    sp_directions,
    z_0,
)
from differt.geometry import (
    Paths,
    cartesian_to_spherical,
    normalize,
    perpendicular_vectors,
)
from differt.plotting import PlotOutput, draw_paths, reuse
from differt.scene import TriangleScene
from differt.utils import dot, safe_divide

ArrayType = TypeVar("ArrayType", bound=Array | np.ndarray)  # type: ignore[reportMissingTypeArgument]
"""Type variable for arrays in the DeepMIMO class."""


class DeepMIMO(eqx.Module, Generic[ArrayType]):
    """DeepMIMO format data structure.

    The generic type parameter ``ArrayType`` can be either a JAX array or a NumPy array, indicating
    whether the data is in JAX or NumPy format.

    For elevation and azimuth angles, the reference coordinate system is detailed in :ref:`conventions`.

    .. todo::

        Avoid :class:`ArrayType` to be documented as an union of JAX and NumPy arrays, but rather as a type variable that can be either JAX or NumPy.
    """

    _: KW_ONLY
    power: Float[ArrayType, "num_tx num_rx num_paths"]
    r"""Tap power. Received power in dBW for each path, assuming 0 dBW transmitted power. :math:`10\log10(|a|^2)`, where :math:`a` is the complex channel amplitude"""
    phase: Float[ArrayType, "num_tx num_rx num_paths"]
    r"""Tap phase. Phase of received signal for each path in degrees. :math:`\angle a` (angle of :math:`a`), where :math:`a` is the complex channel amplitude"""
    delay: Float[ArrayType, "num_tx num_rx num_paths"]
    """Tap delay. Propagation delay for each path in seconds."""
    aoa_az: Float[ArrayType, "num_tx num_rx num_paths"]
    """Angle of arrival (azimuth) for each path in degrees."""
    aoa_el: Float[ArrayType, "num_tx num_rx num_paths"]
    """Angle of arrival (elevation) for each path in degrees."""
    aod_az: Float[ArrayType, "num_tx num_rx num_paths"]
    """Angle of departure (azimuth) for each path in degrees."""
    aod_el: Float[ArrayType, "num_tx num_rx num_paths"]
    """Angle of departure (elevation) for each path in degrees."""
    primitives: (
        Int[ArrayType, "num_tx num_rx num_paths max_num_interactions"] | None
    ) = None
    """Indices of primitives hit along each path.

    .. note::

        This field is optional and is provided if ``include_primitives`` is set to ``True`` in the
        :func:`extract` function.

    A value of -1 indicates no primitive hit, i.e., a path that is terminated early.
    """
    inter: Int[ArrayType, "num_tx num_rx num_paths max_num_interactions"]
    """Type of interactions along each path.

    Matches :class:`InteractionType<differt.em.InteractionType>`, but a value of -1 indicates no interaction,
    i.e., a path that is terminated early.
    """
    inter_pos: Float[ArrayType, "num_rx num_paths max_num_interactions 3"]
    """3D coordinates in meters of each interaction point along paths."""
    rx_pos: Float[ArrayType, "num_rx 3"]
    """Receiver positions in 3D coordinates in meters"""
    tx_pos: Float[ArrayType, "num_tx 3"]
    """Transmitter positions in 3D coordinates in meters"""
    mask: Bool[ArrayType, "num_tx num_rx num_paths"]
    """Mask indicating valid paths."""

    @property
    def num_tx(self) -> int:
        """Number of transmitters."""
        return self.mask.shape[0]

    @property
    def num_rx(self) -> int:
        """Number of receivers."""
        return self.mask.shape[1]

    @property
    def num_paths(self) -> int:
        """Number of paths."""
        return self.mask.shape[2]

    def asdict(self) -> dict[str, ArrayType]:
        """
        Convert this class to a dictionary.

        Returns:
            A dictionary where keys are attribute names and values are arrays.
        """
        return asdict(self)

    def _sort(
        self, primitives: Int[ArrayType, "num_tx num_rx num_paths num_max_interactions"]
    ) -> "DeepMIMO[ArrayType]":
        """Utility function to sort the DeepMIMO based on another object, e.g., :class:`sionna.rt.Paths`."""  # noqa: DOC201, DOC501
        if (
            self.primitives is None or primitives.shape != self.primitives.shape
        ):  # pragma: no cover
            msg = "Cannot sort based on primitives: shape mismatch."
            raise ValueError(msg)

        indices = (
            (
                self.primitives.reshape(-1, 1, self.inter.shape[-1])
                == primitives.reshape(1, -1, self.inter.shape[-1])
            )
            .all(axis=-1)
            .argmax(axis=-1)
        )

        shape_prefix = (self.num_tx, self.num_rx, self.num_paths)

        def sort_fn(x: ArrayType) -> ArrayType:
            if x.shape[:3] != shape_prefix:
                return x

            y = x.reshape(-1, *x.shape[3:])
            y = y[indices, ...]
            return y.reshape(x.shape)

        return jax.tree.map(sort_fn, self)

    def jax(self) -> "DeepMIMO[Array]":
        """
        Return a copy of this class with arrays converted to JAX arrays.

        Returns:
            A copy of this with JAX arrays.
        """
        return jax.tree.map(jnp.asarray, self)

    def numpy(self) -> "DeepMIMO[np.ndarray]":  # type: ignore[reportMissingTypeArgument]
        """
        Return a copy of this class with arrays converted to NumPy arrays.

        Returns:
            A copy of this with NumPy arrays.
        """
        return jax.tree.map(np.asarray, self)

    def plot_paths(self, **kwargs: Any) -> PlotOutput:
        """
        Plot the valid paths in this DeepMIMO object.

        Args:
            kwargs: Keyword arguments passed to
                :func:`draw_paths<differt.plotting.draw_paths>`.

        Returns:
            The resulting plot output.

        Examples:
            The following example shows how to call :func:`export` and use
            the :class:`DeepMIMO` object to plot all paths.

            .. plotly::
                :fig-vars: fig

                >>> import equinox as eqx
                >>> from differt.plugins import deepmimo
                >>> from differt.scene import TriangleScene, get_sionna_scene
                >>> from differt.plotting import reuse
                >>>
                >>> file = get_sionna_scene("simple_street_canyon")
                >>> scene = TriangleScene.load_xml(file)
                >>> scene = eqx.tree_at(
                ...     lambda s: s.transmitters, scene, jnp.array([-33.0, 0.0, 32.0])
                ... )
                >>> scene = eqx.tree_at(
                ...     lambda s: s.receivers, scene, jnp.array([33.0, 0.0, 2.0])
                ... )
                >>>
                >>> with reuse(backend="plotly") as fig:
                ...     scene.plot()
                ...     paths = (
                ...         scene.compute_paths(order=order) for order in [0, 1, 2]
                ...     )
                ...     dm = deepmimo.export(paths=paths, scene=scene, frequency=2.4e9)
                ...     dm.plot_paths()
                >>> fig  # doctest: +SKIP
        """
        with reuse(**kwargs, pass_all_kwargs=True) as output:
            max_num_interactions = self.inter.shape[-1]
            num_interactions = jnp.min(
                jnp.broadcast_to(jnp.arange(max_num_interactions), self.inter.shape),
                initial=max_num_interactions,
                where=self.inter == -1,
                axis=-1,
            )

            for num in range(max_num_interactions + 1):
                where = (self.mask & (num_interactions == num)).reshape(-1)
                tx_pos = jnp.broadcast_to(
                    self.tx_pos[:, None, None, :],
                    (self.num_tx, self.num_rx, self.num_paths, 3),
                ).reshape(-1, 3)[where, :]
                rx_pos = jnp.broadcast_to(
                    self.rx_pos[None, :, None, :],
                    (self.num_tx, self.num_rx, self.num_paths, 3),
                ).reshape(-1, 3)[where, :]
                inter_pos = self.inter_pos.reshape(-1, max_num_interactions, 3)[
                    where, :num, :
                ]
                paths = jnp.concatenate(
                    (tx_pos[..., None, :], inter_pos, rx_pos[..., None, :]), axis=-2
                )
                draw_paths(
                    paths,
                )

        return output


def export(  # noqa: PLR0915
    *,
    paths: Paths | Iterable[Paths],
    scene: TriangleScene,
    radio_materials: Mapping[str, Material] | None = None,
    frequency: Float[ArrayLike, " "],
    include_primitives: bool = False,
) -> DeepMIMO[Array]:
    """
    Export a Ray Tracing simulation to the DeepMIMO format.

    .. note::
        The current implementation assumes far-field propagation in free space, and isotropic antennas.

    Args:
        paths: The geometrical paths.

            You can provide paths with different numbers of interactions by passing an iterable.

            E.g., the return value of :meth:`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>`.

        scene: The scene that was used to compute the paths.
        radio_materials: The list of materials in the scene.

            If not provided, :data:`materials<differt.em._material.materials>` will be used.
        frequency: The operating frequency (in Hz).
        include_primitives: If ``True``, include the primitive indices in the output.

    Returns:
        The exported DeepMIMO data as JAX arrays.

    Raises:
        ValueError: If the scene does not contain information about face materials.
    """
    if scene.mesh.face_materials is None:
        msg = "Scene must contain information about face materials."
        raise ValueError(msg)
    if radio_materials is None:
        radio_materials = materials

    paths_iter = [paths] if isinstance(paths, Paths) else paths
    del paths

    eta_r = jnp.array([
        materials[mat_name].relative_permittivity(frequency)
        for mat_name in scene.mesh.material_names
    ])
    n_r = jnp.sqrt(eta_r)

    tx_pos = scene.transmitters.reshape(-1, 3)
    num_tx = tx_pos.shape[0]
    rx_pos = scene.receivers.reshape(-1, 3)
    num_rx = rx_pos.shape[0]

    no_interaction = -1  # Placeholder for no interaction

    # Fields array
    fields = jnp.zeros((num_tx, num_rx, 0, 3), dtype=complex)
    # Start segments
    start_segments = jnp.zeros((num_tx, num_rx, 0, 3), dtype=float)
    # End segments
    end_segments = jnp.zeros_like(start_segments)
    # Path lengths
    lengths = jnp.zeros((num_tx, num_rx, 0), dtype=float)
    if include_primitives:
        # Primitive indices
        primitives = jnp.zeros((num_tx, num_rx, 0, 0), dtype=int)
    else:
        primitives = None
    # Interaction types
    inter = jnp.zeros((num_tx, num_rx, 0, 0), dtype=int)
    # Interaction point positions
    inter_pos = jnp.zeros((num_tx, num_rx, 0, 0, 3), dtype=float)
    # Mask for valid paths
    mask = jnp.zeros((num_tx, num_rx, 0), dtype=bool)

    for paths in paths_iter:
        # Reshape any batch of tx and rx positions into the expected shape
        paths = paths.reshape(num_tx, num_rx, -1)  # noqa: PLW2901

        # [num_tx num_rx num_path_candidates order+1 3]
        path_segments = jnp.diff(paths.vertices, axis=-2)
        # [num_tx num_rx num_paths 3]
        start_segments = jnp.concatenate(
            (start_segments, path_segments[..., 0, :]), axis=-2
        )
        # [num_tx num_rx num_paths 3]
        end_segments = jnp.concatenate(
            (end_segments, path_segments[..., -1, :]), axis=-2
        )

        max_num_interactions = max(paths.order, inter.shape[-1])
        # [num_tx num_rx num_paths max_num_interactions]
        if primitives is not None:
            primitives = jnp.concatenate(
                (
                    jnp.concatenate(
                        (
                            primitives,
                            jnp.full(
                                (
                                    *primitives.shape[:-1],
                                    max_num_interactions - primitives.shape[-1],
                                ),
                                no_interaction,
                                dtype=primitives.dtype,
                            ),
                        ),
                        axis=-1,
                    ),
                    jnp.concatenate(
                        (
                            paths.objects[..., 1:-1],
                            jnp.full(
                                (
                                    *paths.objects.shape[:-1],
                                    max_num_interactions - paths.order,
                                ),
                                no_interaction,
                                dtype=primitives.dtype,
                            ),
                        ),
                        axis=-1,
                    ),
                ),
                axis=-2,
            )
        # [num_tx num_rx num_paths max_num_interactions]
        inter = jnp.concatenate(
            (
                jnp.concatenate(
                    (
                        inter,
                        jnp.full(
                            (*inter.shape[:-1], max_num_interactions - inter.shape[-1]),
                            no_interaction,
                            dtype=inter.dtype,
                        ),
                    ),
                    axis=-1,
                ),
                jnp.concatenate(
                    (
                        paths.interaction_types
                        if paths.interaction_types is not None
                        else jnp.full_like(
                            paths.objects[..., 1:-1],
                            InteractionType.REFLECTION,
                            dtype=inter.dtype,
                        ),
                        jnp.full(
                            (
                                *paths.objects.shape[:-1],
                                max_num_interactions - paths.order,
                            ),
                            no_interaction,
                            dtype=inter.dtype,
                        ),
                    ),
                    axis=-1,
                ),
            ),
            axis=-2,
        )
        # [num_tx num_rx num_paths max_num_interactions 3]
        inter_pos = jnp.concatenate(
            (
                jnp.concatenate(
                    (
                        inter_pos,
                        jnp.zeros(
                            (
                                *inter_pos.shape[:-2],
                                max_num_interactions - inter_pos.shape[-2],
                                3,
                            ),
                            dtype=inter_pos.dtype,
                        ),
                    ),
                    axis=-2,
                ),
                jnp.concatenate(
                    (
                        paths.vertices[..., 1:-1, :],
                        jnp.zeros(
                            (
                                *paths.vertices.shape[:-2],
                                max_num_interactions - paths.order,
                                3,
                            ),
                            dtype=inter_pos.dtype,
                        ),
                    ),
                    axis=-2,
                ),
            ),
            axis=-3,
        )
        # [num_tx num_rx num_path_candidates order+1 3],
        # [num_tx num_rx num_path_candidates order+1 1]
        k, s = normalize(path_segments, keepdims=True)
        # [num_tx num_rx num_path_candidates order 3]
        k_i = k[..., :-1, :]
        k_r = k[..., +1:, :]

        # Dummy isotropic antenna fields
        # [num_tx num_rx num_paths 3]
        fields_i = perpendicular_vectors(path_segments[..., 0, :]).astype(fields.dtype)

        if paths.order > 0:
            # [num_tx num_rx num_path_candidates order]
            obj_indices = paths.objects[..., 1:-1]
            # [num_tx num_rx num_path_candidates order]
            mat_indices = jnp.take(scene.mesh.face_materials, obj_indices, axis=0)
            # [num_tx num_rx num_path_candidates order 3]
            obj_normals = jnp.take(scene.mesh.normals, obj_indices, axis=0)
            # [num_tx num_rx num_path_candidates order]
            obj_n_r = jnp.take(n_r, mat_indices, axis=0)
            # [num_tx num_rx num_path_candidates order 3]
            (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, obj_normals)
            # [num_tx num_rx num_path_candidates order 1]
            cos_theta = dot(obj_normals, -k_i, keepdims=True)
            # [num_tx num_rx num_path_candidates order 1]
            r_s, r_p = reflection_coefficients(obj_n_r[..., None], cos_theta)
            # [num_tx num_rx num_path_candidates 1]
            r_s = jnp.prod(r_s, axis=-2)
            r_p = jnp.prod(r_p, axis=-2)
            # [num_tx num_rx num_path_candidates order 3]
            (e_i_s, e_i_p), (e_r_s, e_r_p) = sp_directions(k_i, k_r, obj_normals)
            # [num_tx num_rx num_path_candidates 1]
            fields_i_s = dot(fields_i, e_i_s[..., 0, :], keepdims=True)
            fields_i_p = dot(fields_i, e_i_p[..., 0, :], keepdims=True)
            # [num_tx num_rx num_path_candidates 1]
            fields_r_s = r_s * fields_i_s
            fields_r_p = r_p * fields_i_p
            # [num_tx num_rx num_path_candidates 3]
            fields_r = fields_r_s * e_r_s[..., -1, :] + fields_r_p * e_r_p[..., -1, :]
        else:
            # [num_tx num_rx num_path_candidates 3]
            fields_r = fields_i

        # [num_tx num_rx num_path_candidates 1]
        s_tot = s.sum(axis=-2)
        spreading_factor = safe_divide(1.0, s_tot)  # Far-field approximation
        wavenumber = 2 * jnp.pi * frequency / c
        phase_shift = jnp.exp(1j * s_tot * wavenumber)
        # [num_tx num_rx num_path_candidates 3]
        fields_r *= spreading_factor * phase_shift

        fields = jnp.concatenate((fields, fields_r), axis=-2)

        # [num_tx num_rx num_paths]
        lengths = jnp.concatenate(
            (lengths, s_tot[..., 0]),
            axis=-1,
        )

        # [num_tx num_rx num_paths]
        mask = jnp.concatenate(
            (
                mask,
                paths.mask
                if paths.mask is not None
                else jnp.ones((num_tx, num_rx, paths.vertices.shape[2]), dtype=bool),
            ),
            axis=-1,
        )

    # TODO: check if this is correct and if we can avoid complex numbers (not using abs)
    power = jnp.abs(dot(fields, fields)) / z_0
    # Scale by the antenna effective aperture
    wavelength = c / frequency
    power *= wavelength**2 / (4 * jnp.pi)

    power = 10 * jnp.log10(power)  # Convert to dBW
    # TODO: we need to project the fields to the antenna's direction (otherwise we don't have scalar values)
    phase = jnp.angle(fields, deg=True)
    delay = lengths / c
    _, aoa_el, aoa_az = jnp.split(
        cartesian_to_spherical(end_segments),
        3,
        axis=-1,
    )
    aoa_az = jnp.rad2deg(aoa_az)
    aoa_el = jnp.rad2deg(aoa_el)
    _, aod_el, aod_az = jnp.split(
        cartesian_to_spherical(start_segments),
        3,
        axis=-1,
    )
    aod_az = jnp.rad2deg(aod_az)
    aod_el = jnp.rad2deg(aod_el)

    return DeepMIMO(
        power=power,
        phase=phase,
        delay=delay,
        aoa_az=aoa_az,
        aoa_el=aoa_el,
        aod_az=aod_az,
        aod_el=aod_el,
        inter=inter,
        inter_pos=inter_pos,
        rx_pos=rx_pos,
        tx_pos=tx_pos,
        mask=mask,
        primitives=primitives,
    )

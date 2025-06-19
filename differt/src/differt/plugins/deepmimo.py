"""Utilities to work with the DeepMIMO framework."""

__all__ = ("DeepMIMO", "export")

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import KW_ONLY, asdict
from typing import Any, Generic

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
)
from differt.plotting import PlotOutput, draw_paths, reuse
from differt.rt import SizedIterator
from differt.scene import TriangleScene
from differt.utils import dot, safe_divide

from ._deepmimo_types import ArrayType


class DeepMIMO(eqx.Module, Generic[ArrayType]):
    """DeepMIMO format data structure.

    The generic type parameter ``ArrayType`` can be either a JAX array or a NumPy array, indicating
    whether the data is in JAX or NumPy format.

    For elevation and azimuth angles, the reference coordinate system is detailed in :ref:`conventions`.
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

        This field is optional and is provided if ``include_primitives`` is set to :data:`True` in the
        :func:`export` function.

    A value of -1 indicates no primitive hit, i.e., a path that is terminated early.
    """
    inter: Int[ArrayType, "num_tx num_rx num_paths max_num_interactions"]
    """Type of interactions along each path.

    Matches :class:`InteractionType<differt.em.InteractionType>`, but a value of -1 indicates no interaction,
    i.e., a path that is terminated early.
    """
    inter_pos: Float[ArrayType, "num_tx num_rx num_paths max_num_interactions 3"]
    """3D coordinates in meters of each interaction point along paths."""
    rx_pos: Float[ArrayType, "num_rx 3"]
    """Receiver positions in 3D coordinates in meters."""
    tx_pos: Float[ArrayType, "num_tx 3"]
    """Transmitter positions in 3D coordinates in meters."""
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
        self,
        paths: "sionna.rt.Paths",  # type: ignore[reportUndefinedVariable]  # noqa: F821
    ) -> "DeepMIMO[ArrayType]":
        """Utility function to sort the DeepMIMO based on :class:`sionna.rt.Paths`' vertices."""  # noqa: DOC201, DOC501
        vertices = jnp.moveaxis(paths.vertices.jax(), 0, -2)
        interactions = (
            jnp.moveaxis(paths.interactions.jax(), 0, -1).astype(self.inter.dtype) - 1
        )

        if vertices.shape != self.inter_pos.shape:  # pragma: no cover
            msg = "Cannot sort based on provided paths: shape mismatch, got {vertices.shape!r} but expected {self.inter_pos.shape!r}."
            raise ValueError(msg)

        max_num_interactions = self.inter.shape[-1]
        indices = (
            jnp.linalg.norm(
                self.inter_pos.reshape(-1, 1, max_num_interactions, 3)
                - vertices.reshape(
                    1,
                    -1,
                    max_num_interactions,
                    3,
                ),
                axis=3,
            )
            .sum(
                axis=2,
                initial=jnp.where(
                    (
                        self.inter.reshape(-1, 1, max_num_interactions)
                        == interactions.reshape(1, -1, max_num_interactions)
                    ).all(axis=-1),
                    jnp.inf,
                    0,
                ),
                where=self.inter.reshape(-1, 1, max_num_interactions) != -1,
            )
            .argmin(axis=1)
        )

        shape_prefix = (self.num_tx, self.num_rx, self.num_paths)

        def sort_fn(x: ArrayType) -> ArrayType:
            if x.shape[: len(shape_prefix)] != shape_prefix:
                return x

            y = x.reshape(-1, *x.shape[len(shape_prefix) :])
            y = y[indices, ...]
            return y.reshape(x.shape)  # type: ignore[reportReturnType]

        return jax.tree.map(sort_fn, self)

    def jax(self) -> "DeepMIMO[Array]":
        """
        Return a copy of this class with arrays converted to JAX arrays.

        Returns:
            A copy of this with JAX arrays.
        """
        return jax.tree.map(jnp.asarray, self)

    def numpy(self) -> "DeepMIMO[np.ndarray]":
        """
        Return a copy of this class with arrays converted to NumPy arrays.

        Returns:
            A copy of this with NumPy arrays.
        """
        return jax.tree.map(np.asarray, self)

    def iter_paths(
        self,
    ) -> SizedIterator[Float[Array, "num_tx num_rx num_paths num_interactions 3"]]:
        """
        Return an iterator over the path vertices in this DeepMIMO object.

        Returns:
            An iterator of path vertices, grouped by ascending number of interactions, from
            ``0`` to ``max_num_interactions``.
        """
        max_num_interactions = self.inter.shape[-1]

        def it() -> Iterator[
            Float[Array, "num_tx num_rx num_paths num_interactions 3"]
        ]:
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
                yield jnp.concatenate(
                    (tx_pos[..., None, :], inter_pos, rx_pos[..., None, :]), axis=-2
                )

        return SizedIterator(it(), size=max_num_interactions + 1)

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
                >>> with reuse(backend="plotly") as fig:  # doctest: +SKIP
                ...     scene.plot()
                ...     paths = (
                ...         scene.compute_paths(order=order) for order in [0, 1, 2]
                ...     )
                ...     dm = deepmimo.export(paths=paths, scene=scene, frequency=2.4e9)
                ...     dm.plot_paths()
                >>> fig  # doctest: +SKIP
        """
        with reuse(**kwargs, pass_all_kwargs=True) as output:
            for paths in self.iter_paths():
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
        The current implementation assumes far-field propagation in free space, and isotropic antennas with vertical polarization.

        While tests show a good match with Sionna's :class:`sionna.rt.PathSolver` for most attributes,
        :attr:`DeepMIMO.power` and :attr:`DeepMIMO.phase` are not exactly equals, and we don't know yet if our implementation is 100% correct. If you know how to improve this, please open an issue or a pull-request on GitHub!

    Args:
        paths: The geometrical paths.

            You can provide paths with different numbers of interactions by passing an iterable.

            E.g., the return value of :meth:`TriangleScene.compute_paths<differt.scene.TriangleScene.compute_paths>`.

        scene: The scene that was used to compute the paths.
        radio_materials: The list of materials in the scene.

            If not provided, :data:`materials<differt.em._material.materials>` will be used.
        frequency: The operating frequency (in Hz).
        include_primitives: If :data:`True`, include the primitive indices in the output.

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
    polarization = jnp.array([0.0, 0.0, 1.0], dtype=float)
    # Direction of departure (DoD) and direction of arrival (DoA) segments
    k_d = jnp.zeros((num_tx, num_rx, 0, 3), dtype=float)
    k_a = jnp.zeros_like(k_d)
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
        # [num_tx num_rx num_paths 3]
        k_d = jnp.concatenate((k_d, k[..., 0, :]), axis=-2)
        k_a = jnp.concatenate((k_a, -k[..., -1, :]), axis=-2)

        # Dummy isotropic antenna fields
        # [num_tx num_rx num_paths 3]
        fields_i = polarization.astype(fields.dtype)

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
        phase = -2 * jnp.pi * frequency * s_tot / c
        phase_shift = jax.lax.complex(jnp.cos(phase), jnp.sin(phase))
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

    a = dot(fields, polarization)
    wavelength = c / frequency
    a *= wavelength / (4 * jnp.pi)
    power = jnp.abs(a) ** 2 / z_0
    power = 10 * jnp.log10(power)  # Convert to dBW
    phase = jnp.angle(a, deg=True)

    delay = lengths / c
    _, aoa_el, aoa_az = jnp.split(
        cartesian_to_spherical(k_a),
        3,
        axis=-1,
    )
    aoa_az = jnp.rad2deg(aoa_az).squeeze(axis=-1)
    aoa_el = jnp.rad2deg(aoa_el).squeeze(axis=-1)
    _, aod_el, aod_az = jnp.split(
        cartesian_to_spherical(k_d),
        3,
        axis=-1,
    )
    aod_az = jnp.rad2deg(aod_az).squeeze(axis=-1)
    aod_el = jnp.rad2deg(aod_el).squeeze(axis=-1)

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

"""Utilities to work with the DeepMIMO framework."""

__all__ = ("DeepMIMO", "export")

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import KW_ONLY, asdict
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeGuard

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Complex, Float, Int, Shaped

from differt.em import (
    GeometricFieldSolver,
    InteractionType,
    Material,
    compute_cir,
    compute_received_fields,
    compute_received_power,
    materials,
)
from differt.geometry import (
    Mesh,
    Scene,
    SizedIterator,
    TracedPaths,
    cartesian_to_spherical,
    normalize,
)
from differt.plotting import PlotOutput, draw_paths, reuse

from ._deepmimo_types import ArrayType

if TYPE_CHECKING:
    try:
        import sionna.rt
    except ImportError:
        SionnaPaths = Any
    else:
        SionnaPaths = sionna.rt.Paths
else:
    SionnaPaths = Any


def _pad_and_concat(
    left: Shaped[Array, "num_tx num_rx num_paths_left num_interactions_left ..."],
    right: Shaped[Array, "num_tx num_rx num_paths_right num_interactions_right ..."],
    fill_value: Any,
) -> Shaped[
    Array,
    "num_tx num_rx num_paths_left+num_paths_right max(num_interactions_left,num_interactions_right) ...",
]:
    max_num_interactions = max(left.shape[3], right.shape[3])
    extra_dims_pad = [(0, 0)] * (left.ndim - 4)
    left = jnp.pad(
        left,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, max_num_interactions - left.shape[3]),
            *extra_dims_pad,
        ),
        constant_values=fill_value,
    )
    right = jnp.pad(
        right,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, max_num_interactions - right.shape[3]),
            *extra_dims_pad,
        ),
        constant_values=fill_value,
    )
    return jnp.concatenate((left, right), axis=2)


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
        paths: SionnaPaths,
    ) -> "DeepMIMO[Array]":
        """Utility function to sort the DeepMIMO based on :class:`sionna.rt.Paths`' vertices."""  # ruff:ignore[docstring-missing-returns, docstring-missing-exception]
        if _is_jax_dtype(self):
            vertices = jnp.moveaxis(paths.vertices.jax(), 0, -2)
            interactions = (
                jnp.moveaxis(paths.interactions.jax(), 0, -1).astype(self.inter.dtype)
                - 1
            )

            if (
                vertices.shape[:2] != self.inter_pos.shape[:2]
                or vertices.shape[-1] != self.inter_pos.shape[-1]
            ):  # pragma: no cover
                msg = (
                    "Cannot sort based on provided paths: shape mismatch, got "
                    f"{vertices.shape!r} but expected {self.inter_pos.shape!r}."
                )
                raise ValueError(msg)

            max_num_interactions = max(self.inter.shape[-1], vertices.shape[-2])
            inter_pos = jnp.pad(
                self.inter_pos,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, max_num_interactions - self.inter_pos.shape[-2]),
                    (0, 0),
                ),
            )
            inter = jnp.pad(
                self.inter,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, max_num_interactions - self.inter.shape[-1]),
                ),
                constant_values=-1,
            )
            vertices_pad = jnp.pad(
                vertices,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, max_num_interactions - vertices.shape[-2]),
                    (0, 0),
                ),
            )
            interactions_pad = jnp.pad(
                interactions,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, max_num_interactions - interactions.shape[-1]),
                ),
                constant_values=-1,
            )

            # For each of our own paths (axis 0) and each Sionna path (axis 1),
            # sum the vertex-position distance over our own valid interactions.
            distances = jnp.linalg.norm(
                inter_pos.reshape(-1, 1, max_num_interactions, 3)
                - vertices_pad.reshape(
                    1,
                    -1,
                    max_num_interactions,
                    3,
                ),
                axis=3,
            ).sum(
                axis=2,
                where=inter.reshape(-1, 1, max_num_interactions) != -1,
            )
            # Pairs whose interaction pattern doesn't match are disqualified
            # (infinite distance), so that only structurally-compatible paths
            # can be matched based on their (real) position distance. This must
            # be applied after the sum (rather than via its 'initial' argument)
            # because 'jnp.sum' silently ignores 'initial' when 'where' excludes
            # every element being reduced, e.g., for line-of-sight paths.
            matches = (
                inter.reshape(-1, 1, max_num_interactions)
                == interactions_pad.reshape(1, -1, max_num_interactions)
            ).all(axis=-1)
            distances = jnp.where(matches, distances, jnp.inf)
            # For each Sionna path, find the index of our own closest-matching
            # path (not the other way around), so that gathering with 'indices'
            # below correctly reorders our paths into Sionna's order, even when
            # the correspondence between the two orderings is not an involution.
            indices = distances.argmin(axis=0)

            orig_shape_prefix = (self.num_tx, self.num_rx, self.num_paths)
            target_shape_prefix = (*vertices.shape[:3],)

            def sort_fn(x: Array) -> Array:
                if (
                    x is None
                    or not isinstance(x, (np.ndarray, jax.Array))
                    or x.shape[: len(orig_shape_prefix)] != orig_shape_prefix
                ):
                    return x

                y = x.reshape(-1, *x.shape[len(orig_shape_prefix) :])
                y = y[indices, ...]
                return y.reshape(
                    *target_shape_prefix, *x.shape[len(orig_shape_prefix) :]
                )

            return jax.tree.map(sort_fn, self)

        return self.jax()._sort(paths)  # ruff:ignore[private-member-access]

    def jax(self) -> "DeepMIMO[Array]":
        """
        Return a new instance of this class with arrays converted to JAX arrays.

        Returns:
            A new instance of this with JAX arrays.
        """
        return jax.tree.map(jnp.asarray, self)

    def numpy(self) -> "DeepMIMO[np.ndarray]":
        """
        Return a new instance of this class with arrays converted to NumPy arrays.

        Returns:
            A new instance of this with NumPy arrays.
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
        # TODO: test this method
        if _is_jax_dtype(self):
            max_num_interactions = self.inter.shape[-1]

            def it() -> Iterator[
                Float[Array, "num_tx num_rx num_paths num_interactions 3"]
            ]:
                num_interactions = jnp.min(
                    jnp.broadcast_to(
                        jnp.arange(max_num_interactions), self.inter.shape
                    ),
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
        return self.jax().iter_paths()

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
                >>> from differt.geometry import Scene, get_sionna_scene
                >>> from differt.plotting import reuse
                >>>
                >>> file = get_sionna_scene("simple_street_canyon")
                >>> scene = Scene.load_xml(file)
                >>> scene = eqx.tree_at(
                ...     lambda s: s.transmitters, scene, jnp.array([-33.0, 0.0, 32.0])
                ... )
                >>> scene = eqx.tree_at(
                ...     lambda s: s.receivers, scene, jnp.array([33.0, 0.0, 2.0])
                ... )
                >>>
                >>> with reuse(backend="plotly") as fig:  # doctest: +SKIP
                ...     scene.plot()
                ...     paths = (scene.trace_paths(order=order) for order in [0, 1, 2])
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


def _process_chunk(
    paths: TracedPaths,
    mesh: Mesh,
    frequency: Float[ArrayLike, ""],
    solver: GeometricFieldSolver,
) -> tuple[
    Int[Array, "num_tx num_rx num_paths order"],
    Int[Array, "num_tx num_rx num_paths order"],
    Float[Array, "num_tx num_rx num_paths order 3"],
    Float[Array, "num_tx num_rx num_paths 3"],
    Float[Array, "num_tx num_rx num_paths 3"],
    Complex[Array, "num_tx num_rx num_paths"],
    Float[Array, "num_tx num_rx num_paths"],
    Bool[Array, "num_tx num_rx num_paths"],
]:
    """
    Compute per-path electromagnetic quantities for one batch (chunk) of same-order paths.

    This is factored out of :func:`export` for readability. It is deliberately not
    JIT-compiled itself, since ``solver.radio_materials`` is a plain (unhashable)
    mapping and cannot be part of a JIT-traced pytree; the actual numeric work is
    still performed by the already JIT-compiled
    :func:`compute_received_fields<differt.em.compute_received_fields>` and
    :func:`compute_cir<differt.em.compute_cir>`.

    Returns:
        A tuple containing, in order: the object indices, interaction types,
        interaction positions, DoD unit vectors, DoA unit vectors, channel
        coefficients, path delays, and validity mask, all for this chunk only.
    """
    # [num_tx num_rx num_paths order+1 3]
    path_segments = jnp.diff(paths.vertices, axis=-2)
    k, _ = normalize(path_segments, keepdims=True)

    # [num_tx num_rx num_paths]
    fields_chunk = compute_received_fields(paths, mesh, frequency, solver=solver)
    delay_chunk, fields_chunk = compute_cir(paths, fields_chunk)

    return (
        paths.objects[..., 1:-1],
        paths.interaction_types,
        paths.vertices[..., 1:-1, :],
        k[..., 0, :],
        -k[..., -1, :],
        fields_chunk,
        delay_chunk,
        paths.mask,
    )


def export(
    *,
    paths: TracedPaths | Iterable[TracedPaths],
    scene: Scene,
    radio_materials: Mapping[str, Material] | None = None,
    frequency: Float[ArrayLike, ""],
    include_primitives: bool = False,
    polarization: (
        Literal["V", "H"]
        | Float[ArrayLike, "3"]
        | tuple[
            Literal["V", "H"] | Float[ArrayLike, "3"],
            Literal["V", "H"] | Float[ArrayLike, "3"],
        ]
    ) = "V",
) -> DeepMIMO[Array]:
    """
    Export a Ray Tracing simulation to the DeepMIMO format.

    .. note::
        The current implementation assumes far-field propagation in free space and isotropic antennas.

    Args:
        paths: The geometrical paths.

            You can provide paths with different numbers of interactions by passing an iterable.

            E.g., the return value of :meth:`Scene.trace_paths<differt.geometry.Scene.trace_paths>`.

        scene: The scene that was used to compute the paths.
        radio_materials: The list of materials in the scene.

            If not provided, :data:`materials<differt.em._material.materials>` will be used.
        frequency: The operating frequency (in Hz).
        include_primitives: If :data:`True`, include the primitive indices in the output.
        polarization: The antennas polarization.
            Can be either ``"V"`` (vertical, z-axis up), ``"H"`` (horizontal, x-axis up),
            a 3D unit vector, or a tuple of ``(tx_polarization, rx_polarization)``.

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

    if isinstance(polarization, tuple) and len(polarization) == 2:  # ruff:ignore[magic-value-comparison]
        tx_polarization, rx_polarization = polarization
    else:
        tx_polarization = rx_polarization = polarization

    solver = GeometricFieldSolver(
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        radio_materials=radio_materials,
    )

    paths_iter = [paths] if isinstance(paths, TracedPaths) else paths
    del paths

    tx_pos = scene.transmitters.reshape(-1, 3)
    num_tx = tx_pos.shape[0]
    rx_pos = scene.receivers.reshape(-1, 3)
    num_rx = rx_pos.shape[0]

    # Channel coefficients (received complex fields) array
    a_all = jnp.zeros((num_tx, num_rx, 0), dtype=complex)

    # Direction of departure (DoD) and direction of arrival (DoA) segments
    k_d = jnp.zeros((num_tx, num_rx, 0, 3), dtype=float)
    k_a = jnp.zeros_like(k_d)
    # Path delays
    delay = jnp.zeros((num_tx, num_rx, 0), dtype=float)
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
        paths = paths.reshape(num_tx, num_rx, -1)  # ruff:ignore[redefined-loop-name]

        (
            objects_slice,
            inter_slice,
            inter_pos_slice,
            k_d_chunk,
            k_a_chunk,
            fields_chunk,
            delay_chunk,
            mask_chunk,
        ) = _process_chunk(paths, scene.mesh, frequency, solver)

        # [num_tx num_rx num_paths max_num_interactions]
        if primitives is not None:
            primitives = _pad_and_concat(
                primitives,
                objects_slice,
                fill_value=InteractionType.NONE,
            )
        # [num_tx num_rx num_paths max_num_interactions]
        inter = _pad_and_concat(
            inter,
            inter_slice,
            fill_value=InteractionType.NONE,
        )
        # [num_tx num_rx num_paths max_num_interactions 3]
        inter_pos = _pad_and_concat(
            inter_pos,
            inter_pos_slice,
            fill_value=0.0,
        )
        # [num_tx num_rx num_paths 3]
        k_d = jnp.concatenate((k_d, k_d_chunk), axis=-2)
        k_a = jnp.concatenate((k_a, k_a_chunk), axis=-2)

        a_all = jnp.concatenate((a_all, fields_chunk), axis=-1)

        # [num_tx num_rx num_paths]
        delay = jnp.concatenate((delay, delay_chunk), axis=-1)

        # [num_tx num_rx num_paths]
        mask = jnp.concatenate((mask, mask_chunk), axis=-1)

    power = compute_received_power(a_all)
    phase = jnp.angle(a_all, deg=True)

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


def _is_jax_dtype(dm: DeepMIMO[ArrayType]) -> TypeGuard[DeepMIMO[Array]]:
    return isinstance(dm.power, Array)

# ruff:file-ignore[math-constant]

import typing
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

if TYPE_CHECKING or hasattr(typing, "GENERATING_DOCS"):
    from typing import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


class Material(eqx.Module):
    """A class representing a material and it electrical properties."""

    name: str = eqx.field(static=True)
    """
    The name of the material.
    """
    properties: Callable[
        [Float[ArrayLike, " *batch"]],
        tuple[Float[Array, " *batch"], Float[Array, " *batch"]],
    ] = eqx.field(static=True)
    """
    The callable that computes the electrical properties of the material at the given frequency.

    The signature of the callable must be as follows.

    Args:
        frequency: The frequency at which to compute the electrical properties.

    Returns:
        A tuple containing the relative permittivity and conductivity of the material.
    """
    thickness: Float[ArrayLike, ""] | None = eqx.field(default=None)
    """The thickness of the material."""
    aliases: tuple[str, ...] = eqx.field(default=(), static=True)
    """
    A tuple of name aliases for the material.
    """
    _itu_properties: (
        tuple[
            tuple[
                Any,
                Any,
                Any,
                Any,
                tuple[Any, Any] | None,
            ],
            ...,
        ]
        | None
    ) = eqx.field(default=None, static=True)
    """
    ITU frequency-dependent parameters if constructed via :meth:`from_itu_properties`.
    """

    def __repr__(self) -> str:
        thickness_str = (
            f", thickness={self.thickness!r}" if self.thickness is not None else ""
        )
        aliases_str = f", aliases={self.aliases!r}" if self.aliases else ""
        return f"Material(name={self.name!r}{thickness_str}{aliases_str})"

    def relative_permittivity(
        self, frequency: Float[ArrayLike, " *batch"]
    ) -> Float[Array, " *batch"]:
        """
        Compute the relative permittivity of the material at the given frequency.

        Args:
            frequency: The frequency at which to compute the relative permittivity.

        Returns:
            The relative permittivity of the material.
        """
        return self.properties(frequency)[0]

    def conductivity(
        self, frequency: Float[ArrayLike, " *batch"]
    ) -> Float[Array, " *batch"]:
        """
        Compute the conductivity of the material at the given frequency.

        Args:
            frequency: The frequency at which to compute the conductivity.

        Returns:
            The conductivity of the material.
        """
        return self.properties(frequency)[1]

    @classmethod
    def from_itu_properties(
        cls,
        name: str,
        *itu_properties: tuple[
            Float[ArrayLike, ""],
            Float[ArrayLike, ""],
            Float[ArrayLike, ""],
            Float[ArrayLike, ""],
            tuple[Float[ArrayLike, ""], Float[ArrayLike, ""]] | None,
        ],
    ) -> Self:
        r"""
        Create a material from ITU properties.

        The ITU-R Recommendation P.2040-4 :cite:`itu-r-2040` defines the electrical properties of a material
        using 4 real-valued coefficients: **a**, **b**, **c**, and **c**. The :data:`materials` mapping
        is already populated with values from :cite:`itu-r-2040{Tab. 3}`.

        Args:
            name: The name of the material.
            itu_properties: Material properties and corresponding frequency range.

                Each tuple must contain:

                * **a** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The first coefficient for the real part of the relative permittivity.
                * **b** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The second coefficient for the real part of the relative permittivity.
                * **c** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The first coefficient for the conductivity.
                * **d** (:class:`Float[ArrayLike, '']<jaxtyping.Float>`):
                  The second coefficient for the conductivity.
                * **frequency_range**
                  (:class:`tuple`\[:class:`Float[ArrayLike, '']<jaxtyping.Float>`,
                  :class:`Float[ArrayLike, '']<jaxtyping.Float>`\]):
                  The frequency range (in GHz) for which the electrical
                  properties are assumed to be correct.

        Returns:
            The material with the given ITU properties.

        Raises:
            ValueError: If more than one frequency range is specified and one of them is ``None``.
        """
        if len(itu_properties) > 1 and any(prop[4] is None for prop in itu_properties):
            msg = "Only one frequency range can be used if 'None' is passed, as it will match any frequency"
            raise ValueError(msg)

        aliases = (f"itu_{name.lower().replace(' ', '_')}",)
        if len(itu_properties) == 1:
            a, b, c, d, f_range = itu_properties[0]

            def callback(
                freq: Float[ArrayLike, " *batch"],
                a: Float[ArrayLike, ""],
                b: Float[ArrayLike, ""],
                c: Float[ArrayLike, ""],
                d: Float[ArrayLike, ""],
                f_min_hz: float | None = f_range[0] * 1e9 if f_range else None,
                f_max_hz: float | None = f_range[1] * 1e9 if f_range else None,
            ) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
                f_arr = jnp.asarray(freq)
                if f_min_hz is not None and f_max_hz is not None:
                    where = (f_min_hz <= f_arr) & (f_arr <= f_max_hz)
                else:
                    where = jnp.ones_like(f_arr, dtype=bool)

                freq_ghz = f_arr * 1e-9
                rel_perm = jnp.where(where, a * (freq_ghz**b), -1.0)
                cond = jnp.where(where, c * (freq_ghz**d), -1.0)
                return rel_perm, cond

            return cls(
                name=name,
                properties=partial(jax.jit(callback), a=a, b=b, c=c, d=d),
                aliases=aliases,
                _itu_properties=itu_properties,  # type: ignore[arg-type]
            )

        def properties(
            frequency: Float[ArrayLike, " *batch"],
        ) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
            f_hz = jnp.asarray(frequency)
            batch = f_hz.shape
            f_hz_flat = f_hz.ravel()
            f_ghz_flat = f_hz_flat * 1e-9

            # Sort ranges by lower bound (in Hz)
            ranges_hz = [
                (prop[4][0] * 1e9, prop[4][1] * 1e9)
                if prop[4] is not None
                else (-jnp.inf, jnp.inf)
                for prop in itu_properties
            ]
            sorted_indices = sorted(
                range(len(ranges_hz)), key=lambda i: (ranges_hz[i][0], ranges_hz[i][1])
            )

            lower_bounds_hz = jnp.array([ranges_hz[i][0] for i in sorted_indices])
            upper_bounds_hz = jnp.array([ranges_hz[i][1] for i in sorted_indices])

            # Generate masks for each frequency range (in Hz)
            masks = (f_hz_flat[:, None] >= lower_bounds_hz[None, :]) & (
                f_hz_flat[:, None] <= upper_bounds_hz[None, :]
            )

            # Pick the first matching range for each frequency
            # If outside all ranges, pick fallback index
            i_outside = len(itu_properties)  # Fallback index
            i_range = jnp.arange(len(sorted_indices))
            indices = jnp.where(
                masks,
                i_range[None, :],
                i_outside,
            )

            # Find first True column index per row, or fallback if none match
            indices = jnp.min(indices, axis=1)

            branches = [
                lambda f, a=prop[0], b=prop[1], c=prop[2], d=prop[3]: (
                    a * (f**b),
                    c * (f**d),
                )
                for prop in [itu_properties[i] for i in sorted_indices]
            ]
            branches.append(
                lambda f: (
                    -jnp.ones_like(f),
                    -jnp.ones_like(f),
                )
            )

            rel_perm, cond = jax.vmap(
                lambda freq, idx: jax.lax.switch(
                    idx,
                    branches,
                    freq,
                ),
            )(f_ghz_flat, indices)

            return rel_perm.reshape(batch), cond.reshape(batch)

        return cls(
            name=name,
            properties=properties,
            aliases=aliases,
            _itu_properties=itu_properties,  # type: ignore[arg-type]
        )


class MaterialsDict(dict[str, Material]):  # ruff: ignore[subclass-builtin]
    """A dictionary subclass mapping material names to material instances with automatic alias support.

    This dictionary stores materials keyed by their primary name (:attr:`Material.name`).
    Indexing, membership checks (``in``), getting, and deletion using any material
    alias (defined in :attr:`Material.aliases`) automatically resolve to the primary material.
    """

    def _get_key(self, key: Any) -> Any:
        if not isinstance(key, str):
            return key
        if super().__contains__(key):
            return key
        for primary_key, material in self.items():
            if key in material.aliases:
                return primary_key
        return key

    def __getitem__(self, key: str) -> Material:
        real_key = self._get_key(key)
        return super().__getitem__(real_key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        real_key = self._get_key(key)
        return super().__contains__(real_key)

    def get(self, key: str, default: Any = None) -> Any:
        real_key = self._get_key(key)
        return super().get(real_key, default)

    def __delitem__(self, key: str) -> None:
        real_key = self._get_key(key)
        super().__delitem__(real_key)

    def pop(self, key: str, *args: Any) -> Any:
        real_key = self._get_key(key)
        return super().pop(real_key, *args)

    def setdefault(self, key: str, default: Any = None) -> Any:
        real_key = self._get_key(key)
        if super().__contains__(real_key):
            return self[real_key]
        self[key] = default
        return default

    def __setitem__(self, key: str, value: Material) -> None:
        real_key = self._get_key(key)
        if super().__contains__(real_key):
            super().__setitem__(real_key, value)
        elif isinstance(value, Material):
            super().__setitem__(value.name, value)
        else:
            super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:
        if args:
            if len(args) > 1:
                msg = f"update expected at most 1 argument, got {len(args)}"
                raise TypeError(msg)
            other = args[0]
            if isinstance(other, Mapping):
                for k, v in other.items():
                    self[k] = v
            elif isinstance(other, Iterable):
                for item in other:
                    if isinstance(item, Material):
                        self[item.name] = item
                    else:
                        k, v = item
                        self[k] = v
            else:
                msg = f"Cannot update MaterialsDict from {type(other)}"
                raise TypeError(msg)
        for k, v in kwargs.items():
            self[k] = v

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(*args, **kwargs)


# ITU-R P.2040-4 materials from Table 3.
_materials = [
    Material.from_itu_properties("Vacuum", (1.0, 0.0, 0.0, 0.0, None)),
    Material.from_itu_properties(
        "Concrete",
        (5.24, 0.0, 0.0462, 0.7822, (1.0, 100.0)),
        (5.17, 0.0, 0.0145, 1.09, (110.0, 330.0)),
    ),
    Material.from_itu_properties(
        "Brick",
        (3.91, 0.0, 0.0238, 0.16, (1.0, 40.0)),
        (3.75, 0.0, 0.038, 0.0, (1.0, 10.0)),
        (3.95, 0.0, 0.0022, 1.33, (100.0, 400.0)),
    ),
    Material.from_itu_properties(
        "Plasterboard",
        (2.94, 0.0, 0.0116, 0.7076, (1.0, 100.0)),
        (2.73, 0.0, 0.0084, 0.94, (100.0, 400.0)),
    ),
    Material.from_itu_properties(
        "Wood",
        (1.99, 0.0, 0.0047, 1.0718, (0.001, 100.0)),
        (1.63, 0.0, 0.0076, 1.002, (100.0, 400.0)),
    ),
    Material.from_itu_properties(
        "Glass",
        (6.27, 0.0, 0.0043, 1.1925, (0.1, 100.0)),
        (6.70, 0.0, 0.0042, 1.15, (100.0, 400.0)),
        (6.01, 0.0, 0.0400, 0.81, (220.0, 450.0)),
    ),
    Material.from_itu_properties(
        "Clear Acrylic",
        (2.57, 0.0, 0.0049, 1.0601, (1.0, 40.0)),
    ),
    Material.from_itu_properties(
        "Ceiling board",
        (1.48, 0.0, 0.0011, 1.1278, (1.0, 100.0)),
        (1.58, 0.0, 0.0014, 1.07, (100.0, 400.0)),
    ),
    Material.from_itu_properties(
        "Chipboard",
        (2.58, 0.0, 0.0217, 0.7800, (1.0, 100.0)),
        (2.16, 0.0, 0.0023, 1.359, (100.0, 200.0)),
    ),
    Material.from_itu_properties(
        "Plywood",
        (2.71, 0.0, 0.33, 0.0, (1.0, 40.0)),
        (1.94, 0.0, 0.0067, 0.9982, (110.0, 330.0)),
        (2.17, 0.0, 0.0063, 1.045, (100.0, 400.0)),
    ),
    Material.from_itu_properties(
        "Marble",
        (7.074, 0.0, 0.0055, 0.9262, (1.0, 60.0)),
        (7.94, 0.0, 0.0001, 1.7330, (110.0, 330.0)),
        (8.62, 0.0, 0.0027, 1.15, (100.0, 400.0)),
    ),
    Material.from_itu_properties(
        "Floorboard",
        (3.66, 0.0, 0.0044, 1.3515, (50.0, 100.0)),
        (5.27, 0.0, 2.22e-17, 7.3413, (220.0, 300.0)),
        (5.27, 0.0, 0.0003, 2.0298, (300.0, 400.0)),
        (5.27, 0.0, 49.8726, 0.0, (400.0, 450.0)),
        (3.1575, 0.0, 0.001675, 1.32775, (100.0, 400.0)),
    ),
    Material.from_itu_properties(
        "Vinyl tile",
        (3.62, 0.0, 0.0051, 0.8422, (1.0, 40.0)),
    ),
    Material.from_itu_properties(
        "Carpet tile",
        (2.08, 0.0, 0.0009, 0.8200, (1.0, 40.0)),
    ),
    Material.from_itu_properties(
        "Asphalt concrete",
        (4.83, 0.0, 0.0108, 1.3969, (1.0, 40.0)),
    ),
    Material.from_itu_properties("Metal", (1.0, 0.0, 1e7, 0.0, (1.0, 100.0))),
    Material.from_itu_properties(
        "Very dry ground", (3.0, 0.0, 0.00015, 2.52, (1.0, 10.0))
    ),
    Material.from_itu_properties(
        "Medium dry ground", (15.0, -0.1, 0.035, 1.63, (1.0, 10.0))
    ),
    Material.from_itu_properties("Wet ground", (30.0, -0.4, 0.15, 1.30, (1.0, 10.0))),
]

materials: MaterialsDict = MaterialsDict(_materials)
"""A dictionary mapping material names and their aliases to corresponding :class:`Material` instances.

For convenience, each material can be accessed either by its official ITU name (e.g., ``'Concrete'``)
or by its Sionna-compatible alias (e.g., ``'itu_concrete'``).

See :ref:`itu-materials-table` for a table of all available ITU radio materials, their aliases, and electrical properties."""

del _materials

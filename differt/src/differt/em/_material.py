# ruff: noqa: FURB152
import operator
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
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
            Float[ArrayLike, " "],
            Float[ArrayLike, " "],
            Float[ArrayLike, " "],
            Float[ArrayLike, " "],
            tuple[Float[ArrayLike, " "], Float[ArrayLike, " "]] | None,
        ],
    ) -> Self:
        r"""
        Create a material from ITU properties.

        The ITU-R Recommendation P.2040-3 :cite:`itu-r-2040` defines the electrical properties of a material
        using 4 real-valued coefficients: **a**, **b**, **c**, and **c**. The :data:`materials` mapping
        is already populated with values from :cite:`itu-r-2040{Tab. 3}`.

        Args:
            name: The name of the material.
            itu_properties: The list of material properties and corresponding frequency range.

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

                  This parameter must either be an ordered 2-tuple of min. and max. frequencies,
                  or can be :data:`None`, in which case only one frequency range is allowed as
                  it will match all frequencies.

        Returns:
            A new material.

        Raises:
            ValueError: If you passed more than one frequency range and at least one was :data:`None`.
        """
        f_ranges = []
        branches = []

        dtype = jnp.result_type(*[x for prop in itu_properties for x in prop[:-1]])

        aliases = ("itu_" + name.lower().replace(" ", "_"),)

        @partial(jax.jit, inline=True, static_argnums=(1, 2, 3, 4))
        def callback(
            f: Float[ArrayLike, " *batch"],
            a: Float[ArrayLike, " "],
            b: Float[ArrayLike, " "],
            c: Float[ArrayLike, " "],
            d: Float[ArrayLike, " "],
        ) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
            f_ghz = jnp.asarray(f) / 1e9

            if b == 0:
                rel_perm = jnp.full_like(f_ghz, a, dtype=dtype)
            else:
                rel_perm = jnp.asarray(a * f_ghz**b, dtype=dtype)

            if d == 0:
                cond = jnp.full_like(f_ghz, c, dtype=dtype)
            else:
                cond = jnp.asarray(c * f_ghz**d, dtype=dtype)

            return rel_perm, cond

        if any(prop[-1] is None for prop in itu_properties):
            if len(itu_properties) != 1:
                msg = "Only one frequency range can be used if 'None' is passed, as it will match any frequency."
                raise ValueError(msg)
            a, b, c, d, _ = itu_properties[0]
            return cls(
                name=name,
                properties=partial(callback, a=a, b=b, c=c, d=d),
                aliases=aliases,
            )

        props = sorted(itu_properties, key=operator.itemgetter(-1))

        for a, b, c, d, f_range in props:
            f_ranges.append(f_range)
            branches.append(partial(callback, a=a, b=b, c=c, d=d))

        # This callbacks is used when frequency is outside of range
        branches.append(
            lambda f: (
                -jnp.ones_like(f, dtype=dtype),
                -jnp.ones_like(f, dtype=dtype),
            )
        )
        i_range = jnp.arange(len(f_ranges))
        i_outside = len(branches) - 1

        # NOTE:
        # Checking f >= f_min_ghz * 1e9
        # leads to more accutate check than
        # doing f / 1e9 >= f_min_ghz,
        # hence we pre-multiply frequency ranges to be in Hz.
        f_ranges = jnp.asarray(f_ranges) * 1e9
        f_min = f_ranges[:, 0]
        f_max = f_ranges[:, 1]

        @jax.jit
        def properties(
            f: Float[ArrayLike, "*batch"],
        ) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
            f = jnp.asarray(f)

            if jnp.ndim(f) == 0:
                where = (f_min <= f) & (f <= f_max)
                indices = jnp.min(i_range, where=where, initial=i_outside)
                return jax.lax.switch(
                    indices,
                    branches,
                    f,
                )

            batch = f.shape
            f = f.ravel()

            where = (f_min <= f[..., None]) & (f[..., None] <= f_max)
            indices = jnp.min(
                jnp.broadcast_to(i_range, where.shape),
                where=where,
                initial=i_outside,
                axis=-1,
            )

            rel_perm, cond = jax.vmap(
                lambda freq, i: jax.lax.switch(
                    i,
                    branches,
                    freq,
                ),
            )(f, indices)

            return rel_perm.reshape(batch), cond.reshape(batch)

        return cls(
            name=name,
            properties=properties,
            aliases=aliases,
        )


# ITU-R P.2024-3 materials from Table 3.
_materials = [
    Material.from_itu_properties("Vacuum", (1.0, 0.0, 0.0, 0.0, None)),
    Material.from_itu_properties("Concrete", (5.24, 0.0, 0.0462, 0.7822, (1.0, 100.0))),
    Material.from_itu_properties("Brick", (3.91, 0.0, 0.0238, 0.16, (1.0, 40.0))),
    Material.from_itu_properties(
        "Plasterboard", (2.73, 0.0, 0.0085, 0.9395, (1.0, 100.0))
    ),
    Material.from_itu_properties("Wood", (1.99, 0.0, 0.0047, 1.0718, (0.001, 100.0))),
    Material.from_itu_properties(
        "Glass",
        (6.31, 0.0, 0.0036, 1.3394, (0.1, 100.0)),
        (5.79, 0.0, 0.0004, 1.658, (220.0, 450.0)),
    ),
    Material.from_itu_properties(
        "Ceiling board",
        (1.48, 0.0, 0.0011, 1.0750, (1.0, 100.0)),
        (1.52, 0.0, 0.0029, 1.029, (220.0, 450.0)),
    ),
    Material.from_itu_properties(
        "Chipboard", (2.58, 0.0, 0.0217, 0.7800, (1.0, 100.0))
    ),
    Material.from_itu_properties(
        "Plywood",
        (
            2.71,
            0.0,
            0.33,
            0.0,
            (1.0, 40.0),
        ),
    ),
    Material.from_itu_properties("Marble", (7.074, 0.0, 0.0055, 0.9262, (1.0, 60.0))),
    Material.from_itu_properties(
        "Floorboard", (3.66, 0.0, 0.0044, 1.3515, (50.0, 100.0))
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

materials: dict[str, Material] = {
    name: material
    for material in _materials
    for name in (material.name, *material.aliases)
}
"""A dictionary mapping material names and corresponding object instances.

Some materials, like ITU-R materials, have aliases to match the naming convention of Sionna."""

del _materials

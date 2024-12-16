# ruff: noqa: FURB152
import operator
import sys
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, ArrayLike, Float, jaxtyped

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@jaxtyped(typechecker=typechecker)
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
    Compute the electrical properties of the material at the given frequency.

    Args:
        frequency: The frequency at which to compute the electrical properties.

    Returns:
        A tuple containing the relative permittivity and conductivity of the material.
    """
    aliases: tuple[str, ...] | None = eqx.field(default=None, static=True)
    """
    A tuple of name aliases for the material.
    """

    @eqx.filter_jit
    def relative_permittivity(
        self, frequency: Float[Array, " *batch"]
    ) -> Float[Array, " *batch"]:
        """
        Compute the relative permittivity of the material at the given frequency.

        Args:
            frequency: The frequency at which to compute the relative permittivity.

        Returns:
            The relative permittivity of the material.
        """
        return self.properties(frequency)[0]

    @eqx.filter_jit
    def conductivity(
        self, frequency: Float[Array, " *batch"]
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
        """
        Create a Material instance from JAX properties.

        Args:
            name: The name of the material.
            itu_properties: The list of material properties and corresponding frequency range.
                Each tuple must contain:
                + a: The first coefficient for the real part of the relative permittivity.
                + b: The second coefficient for the real part of the relative permittivity.
                + c: The first coefficient for the conductivity.
                + d: The second coefficient for the conductivity.
                + frequency_range: The frequency range (in GHz) for which the electrical
                    properties are assumed to be correct.

        Returns:
            A Material instance.

        Raises:
            ValueError: If you passed more than one frequency range and at least one was 'None'.
        """
        f_ranges = []
        branches = []

        if any(prop[-1] is None for prop in itu_properties):
            if len(itu_properties) != 1:
                msg = "Only one frequency range can be used if 'None' is passed, as it will match any frequency."
                raise ValueError(msg)
            a, b, c, d, _ = itu_properties[0]
            props = [(a, b, c, d, (float("-inf"), float("+inf")))]
        else:
            props = [
                (a, b, c, d, tuple(sorted(f_range)))  # type: ignore[reportArgumentType]
                for a, b, c, d, f_range in itu_properties
            ]
            props = sorted(itu_properties, key=operator.itemgetter(-1))

        @partial(jax.jit, static_argnums=(1, 2, 3, 4))
        @jaxtyped(typechecker=typechecker)
        def callback(
            f_ghz: Float[ArrayLike, " *#batch"],
            a: Float[ArrayLike, " *#batch"],
            b: Float[ArrayLike, " *#batch"],
            c: Float[ArrayLike, " *#batch"],
            d: Float[ArrayLike, " *#batch"],
        ) -> tuple[Float[ArrayLike, "*batch"], Float[ArrayLike, "*batch"]]:
            return a if b == 0 else a * f_ghz**b, c if d == 0 else c * f_ghz**d

        for a, b, c, d, f_range in props:
            f_ranges.append(f_range)
            branches.append(partial(callback, a=a, b=b, c=c, d=d))

        # This callbacks is used when frequency is outside of range
        branches.append(
            lambda f_ghz: (
                -jnp.ones_like(f_ghz),
                -jnp.ones_like(f_ghz),
            )
        )
        i_outside = len(branches) - 1

        f_ranges = jnp.asarray(f_ranges)
        f_min = f_ranges[:, 0]
        f_max = f_ranges[:, 1]

        # TODO: jitting this seems to cause issues with precision
        @jax.jit
        def properties(
            f: Float[ArrayLike, "*batch"],
        ) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
            f_ghz = jnp.asarray(f / 1e9)

            i_min = jnp.searchsorted(f_min, f_ghz, side="right", method="compare_all")
            i_max = jnp.searchsorted(f_max, f_ghz, side="left", method="compare_all")

            indices = jnp.where(i_max + 1 == i_min, i_max, i_outside)

            if jnp.ndim(f_ghz) == 0:
                return jax.lax.switch(
                    indices,
                    branches,
                    f_ghz,
                )

            rel_perm, cond = jax.vmap(
                lambda f_ghz, i: jax.lax.switch(
                    i,
                    branches,
                    f_ghz,
                ),
            )(f_ghz.ravel(), indices.ravel())

            return rel_perm.reshape(f_ghz.shape), cond.reshape(f_ghz.shape)

        return cls(
            name=name,
            properties=properties,
            aliases=("itu_" + name.lower().replace(" ", "_"),),
        )


_materials = [
    Material.from_itu_properties("Vacuum", (1.0, 0.0, 0.0, 0.0, None)),
    Material.from_itu_properties("Concrete", (5.24, 0.0, 0.0462, 0.7822, (1, 100))),
    Material.from_itu_properties("Brick", (3.91, 0.0, 0.0238, 0.16, (1, 40))),
    Material.from_itu_properties("Plasterboard", (2.73, 0.0, 0.0085, 0.9395, (1, 100))),
    Material.from_itu_properties("Wood", (1.99, 0.0, 0.0047, 1.0718, (0.001, 100))),
    Material.from_itu_properties(
        "Glass",
        (6.31, 0.0, 0.0036, 1.3394, (0.1, 100)),
        (5.79, 0.0, 0.0004, 1.658, (220, 450)),
    ),
    Material.from_itu_properties(
        "Ceiling board",
        (1.48, 0.0, 0.0011, 1.0750, (1, 100)),
        (1.52, 0.0, 0.0029, 1.029, (220, 450)),
    ),
    Material.from_itu_properties("Chipboard", (2.58, 0.0, 0.0217, 0.7800, (1, 100))),
    Material.from_itu_properties(
        "Plywood",
        (
            2.71,
            0.0,
            0.33,
            0.0,
            (1, 40),
        ),
    ),
    Material.from_itu_properties("Marble", (7.074, 0.0, 0.0055, 0.9262, (1, 60))),
    Material.from_itu_properties("Floorboard", (3.66, 0.0, 0.0044, 1.3515, (50, 100))),
    Material.from_itu_properties("Metal", (1.0, 0.0, 1e7, 0.0, (1, 100))),
    Material.from_itu_properties("Very dry ground", (3.0, 0.0, 0.00015, 2.52, (1, 10))),
    Material.from_itu_properties(
        "Medium dry ground", (15.0, -0.1, 0.035, 1.63, (1, 10))
    ),
    Material.from_itu_properties("Wet ground", (30.0, -0.4, 0.15, 1.30, (1, 10))),
]

materials: dict[str, Material] = {
    name: material
    for material in _materials
    for name in (material.name, *(material.aliases or ()))
}
"""A dictionary mapping material names and corresponding object instances.

Some materials, like ITU-R materials, have aliases to match the naming convention of Sionna."""

del _materials

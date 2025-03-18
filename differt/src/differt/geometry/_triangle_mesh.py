# ruff: noqa: ERA001

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PRNGKeyArray

import differt_core.geometry
from differt.plotting import PlotOutput, draw_mesh

from ._utils import normalize, orthogonal_basis, rotation_matrix_along_axis

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
else:
    Self = Any  # Because runtime type checking from 'beartype' will fail when combined with 'jaxtyping'


@jax.jit
def triangles_contain_vertices_assuming_inside_same_plane(
    triangle_vertices: Float[ArrayLike, "*#batch 3 3"],
    vertices: Float[ArrayLike, "*#batch 3"],
) -> Bool[Array, " *#batch"]:
    """
    Return whether each triangle contains the corresponding vertex, but assuming the vertex lies in the same plane as the triangle.

    This is especially useful when combined with the
    :func:`image_method<differt.rt.image_method>`, as the paths returned
    will also lie in the same plane as the mirrors, but may be outside of the actual reflector,
    e.g., a triangular surface.

    Args:
        triangle_vertices: an array of triangle vertices.
        vertices: an array of vertices that will be checked.

    Returns:
        A boolean array indicating whether vertices are in the corresponding triangles or not.
    """
    triangle_vertices = jnp.asarray(triangle_vertices)
    vertices = jnp.asarray(vertices)

    # [*batch 3]
    p0 = triangle_vertices[..., 0, :]
    p1 = triangle_vertices[..., 1, :]
    p2 = triangle_vertices[..., 2, :]

    # Vectors from test vertex to every triangle vertex
    # [*batch 3]
    u0 = p0 - vertices
    u1 = p1 - vertices
    u2 = p2 - vertices

    # Vectors from one triangle vertex to the next
    # [*batch 3]
    v0 = p1 - p0
    v1 = p2 - p1
    v2 = p0 - p2

    # Cross product between corresponding vectors,
    # resulting 'normal' vector should all be perpendicular
    # to the triangle surface
    # [*batch 3]
    n0 = jnp.cross(u0, v0)
    n1 = jnp.cross(u1, v1)
    n2 = jnp.cross(u2, v2)

    # Dot product between all pairs of 'normal' vectors
    # [*batch]
    d01 = jnp.sum(n0 * n1, axis=-1)
    d12 = jnp.sum(n1 * n2, axis=-1)
    d20 = jnp.sum(n2 * n0, axis=-1)

    # [*batch]
    all_pos = (d01 >= 0.0) & (d12 >= 0.0) & (d20 >= 0.0)
    all_neg = (d01 <= 0.0) & (d12 <= 0.0) & (d20 <= 0.0)

    # The vertices are contained if all signs are the same
    return all_pos | all_neg


class TriangleMesh(eqx.Module):
    """A simple geometry made of triangles."""

    vertices: Float[Array, "num_vertices 3"] = eqx.field(converter=jnp.asarray)
    """The array of triangle vertices."""
    triangles: Int[Array, "num_triangles 3"] = eqx.field(converter=jnp.asarray)
    """The array of triangle indices."""
    face_colors: Float[Array, "num_triangles 3"] | None = eqx.field(
        converter=lambda x: jnp.asarray(x) if x is not None else None, default=None
    )
    """The array of face colors.

    The array contains the face colors, as RGB triplets,
    with a black color used as defaults (if some faces have a color).
    This attribute is :data:`None` if all face colors are unset.
    """
    face_materials: Int[Array, " num_triangles"] | None = eqx.field(
        converter=lambda x: jnp.asarray(x) if x is not None else None, default=None
    )
    """The array of face materials.

    The array contains the material indices,
    with a special placeholder value of ``-1``.
    The obtain the name of the material, see :attr:`material_names`.
    This attribute is :data:`None` if all face materials are unset.
    """
    material_names: tuple[str, ...] = eqx.field(
        converter=tuple, default_factory=tuple, static=True
    )
    """The list of material names (must be unique)."""
    object_bounds: Int[Array, "num_objects 2"] | None = eqx.field(
        converter=lambda x: jnp.asarray(x) if x is not None else None, default=None
    )
    """The array of object indices.

    If the present mesh contains multiple objects, usually as a result of appending
    multiple meshes together, this array contain start end end indices for each sub mesh.

    .. important::

        The object indices must cover exactly all triangles in this mesh,
        and be sorted in ascending order. Otherwise, some methods, like
        the random object coloring with :meth:`set_face_colors`, may not
        work as expected.
    """
    assume_quads: bool = eqx.field(default=False)
    """Flag indicating whether triangles can be paired into quadrilaterals.

    Setting this to :data:`True` will not check anything, except that
    :attr:`num_triangles` is even, but each two consecutive
    triangles are assumed to represent a quadrilateral surface.
    """

    def __check_init__(self) -> None:  # noqa: PLW3201
        if self.assume_quads and (self.triangles.shape[0] % 2) != 0:
            msg = "You cannot set 'assume_quads' to 'True' if the number of triangles is not even!"
            raise ValueError(msg)
        if len(set(self.material_names)) != len(self.material_names):
            msg = f"Material names must be unique, got {self.material_names!r}."
            raise ValueError(msg)

    def __getitem__(self, key: slice | Int[ArrayLike, " n"]) -> Self:
        """Return a copy of this mesh, taking only specific triangles.

        Warning:
            As it is not possible to guarantee that indexing would not break existing
            object bounds, the :attr:`object_bounds` attributed is simply dropped.

        Args:
            key: The key used to index :attr:`triangles`
                along the first axis.

        Returns:
            A new mesh.
        """
        return eqx.tree_at(
            lambda m: (
                m.vertices,
                m.triangles,
                m.face_colors,
                m.face_materials,
                m.object_bounds,
            ),
            self,
            (
                self.vertices,
                self.triangles[key, :],
                self.face_colors[key, :] if self.face_colors is not None else None,
                self.face_materials[key] if self.face_materials is not None else None,
                None,
            ),
            is_leaf=lambda x: x is None,
        )

    def iter_objects(self) -> Iterator[Self]:
        """
        Iterator over sub meshes (i.e., objects) defined by :attr:`object_bounds`.

        If :attr:`object_bounds` is :data:`None`, then yield ``self``.

        Yields:
            One or more sub meshes.
        """
        if self.object_bounds is None:
            yield self
        else:
            for start, stop in self.object_bounds:
                yield self[start:stop]

    @property
    def num_triangles(self) -> int:
        """The number of triangles."""
        return self.triangles.shape[0]

    @property
    def num_quads(self) -> int:
        """The number of quadrilaterals.

        Raises:
            ValueError: If :attr:`assume_quads` is :data:`False`.
        """
        if not self.assume_quads:
            msg = "Cannot access the number of quadrilaterals if 'assume_quads' is set to 'False'."
            raise ValueError(msg)

        return self.triangles.shape[0] // 2

    @property
    def num_objects(self) -> int:
        """The number of objects.

        This is a convenient alias to :attr:`num_quads` :attr:`assume_quads` is :data:`True`
        else :attr:`num_triangles`.
        """
        return self.num_quads if self.assume_quads else self.num_triangles

    @property
    def triangle_vertices(self) -> Float[Array, "num_triangles 3 3"]:
        """The array of indexed triangle vertices."""
        if self.triangles.size == 0:
            return jnp.empty_like(self.vertices, shape=(0, 3, 3))

        return jnp.take(self.vertices, self.triangles, axis=0)

    def set_assume_quads(self, flag: bool = True) -> Self:
        """
        Return a copy of this mesh with :attr:`assume_quads` set to ``flag``.

        Unlike with using :func:`equinox.tree_at`, this function will also
        perform runtime checks.

        Args:
            flag: The new flag value.

        Returns:
            A new mesh.
        """
        mesh = eqx.tree_at(lambda m: m.assume_quads, self, flag)
        mesh.__check_init__()
        return mesh

    @classmethod
    def from_core(cls, core_mesh: differt_core.geometry.TriangleMesh) -> Self:
        """
        Return a triangle mesh from a mesh created by the :mod:`differt_core` module.

        Args:
            core_mesh: The mesh from the core module.

        Returns:
            The corresponding mesh.
        """
        return cls(
            vertices=core_mesh.vertices,
            triangles=core_mesh.triangles.astype(int),
            face_colors=core_mesh.face_colors,
            face_materials=core_mesh.face_materials,
            material_names=tuple(core_mesh.material_names),
            object_bounds=core_mesh.object_bounds.astype(int)
            if core_mesh.object_bounds is not None
            else None,
        )

    @property
    def normals(self) -> Float[Array, "num_triangles 3"]:
        """The triangle normals."""
        vectors = jnp.diff(self.triangle_vertices, axis=1)
        normals = jnp.cross(vectors[:, 0, :], vectors[:, 1, :])

        return normalize(normals)[0]

    @property
    def diffraction_edges(self) -> Int[Array, "num_edges 3"]:
        """The diffraction edges."""
        raise NotImplementedError

    @property
    def bounding_box(self) -> Float[Array, "2 3"]:
        """The bounding box (min. and max. coordinates)."""
        # Using self.triangle_vertices is important because, e.g., as a result of using
        # __getitem__, some vertices in 'self.vertices' may no longer be used by this mesh.
        vertices = self.triangle_vertices.reshape(-1, 3)
        return jnp.vstack(
            (jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)),
        )

    def rotate(self, rotation_matrix: Float[ArrayLike, "3 3"]) -> Self:
        """
        Return a new mesh by applying a rotation matrix to all triangle coordinates.

        Args:
            rotation_matrix: The rotation matrix.

        Returns:
            The new rotated mesh.
        """
        return eqx.tree_at(
            lambda m: m.vertices,
            self,
            (jnp.asarray(rotation_matrix) @ self.vertices.T).T,
        )

    def scale(self, scale_factor: Float[ArrayLike, " "]) -> Self:
        """
        Return a new mesh by applying a scale factor to all triangle coordinates.

        Args:
            scale_factor: The scate factor.

        Returns:
            The new scaled mesh.
        """
        return eqx.tree_at(
            lambda m: m.vertices,
            self,
            self.vertices * scale_factor,
        )

    def translate(self, translation: Float[ArrayLike, "3"]) -> Self:
        """
        Return a new mesh by applying a translation to all triangle coordinates.

        Args:
            translation: The translation vector.

        Returns:
            The new translated mesh.
        """
        return eqx.tree_at(
            lambda m: m.vertices,
            self,
            self.vertices + translation,
        )

    @classmethod
    def empty(cls) -> Self:
        """
        Create a empty mesh.

        Returns:
            A new empty scene.
        """
        return cls(vertices=jnp.empty((0, 3)), triangles=jnp.empty((0, 3), dtype=int))

    def append(self, other: "TriangleMesh") -> "TriangleMesh":
        """
        Return a new mesh by appending another mesh to this one.

        .. tip::

            For convenience, you can also use the ``+`` operator.

        .. todo::

            Document how optional arguments are merged,
            as well as the behavior of :attr:`assume_quads`,
            and the specific case of empty meshes.

        Args:
            other: The mesh to append.

        Returns:
            The new mesh.

        Examples:
            The following example shows how to create a mesh of nested cubes.

            .. plotly::

                >>> from differt.geometry import TriangleMesh
                >>>
                >>> mesh = TriangleMesh.empty()
                >>> for i in range(3):
                ...     size = 1.0 / (i + 1)
                ...     mesh += TriangleMesh.box(length=size, width=size, height=size)
                >>> mesh = mesh.set_assume_quads().set_face_colors(
                ...     key=jax.random.key(1234)
                ... )
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig  # doctest: +SKIP
        """
        if self.is_empty:
            return eqx.tree_at(lambda _: (), other, ())
        if other.is_empty:
            return eqx.tree_at(lambda _: (), self, ())

        vertices = jnp.concatenate((self.vertices, other.vertices), axis=0)
        triangles = jnp.concatenate(
            (self.triangles, other.triangles + self.vertices.shape[0]), axis=0
        )

        if self.face_colors is not None and other.face_colors is not None:
            face_colors = jnp.concatenate((self.face_colors, other.face_colors), axis=0)
        elif self.face_colors is not None:
            face_colors = jnp.concatenate(
                (
                    self.face_colors,
                    jnp.zeros_like(self.face_colors, shape=(other.num_triangles, 3)),
                ),
                axis=0,
            )
        elif other.face_colors is not None:
            face_colors = jnp.concatenate(
                (
                    jnp.zeros_like(other.face_colors, shape=(self.num_triangles, 3)),
                    other.face_colors,
                ),
                axis=0,
            )
        else:
            face_colors = None

        material_names = dict.fromkeys(self.material_names) | dict.fromkeys(
            other.material_names
        )
        material_indices = {
            material_name: i for i, material_name in enumerate(material_names)
        }
        other_face_materials_renumbered = jnp.array([
            material_indices[name] for name in other.material_names
        ])
        material_names = tuple(material_names)

        if self.face_materials is not None and other.face_materials is not None:
            face_materials = jnp.concatenate(
                (
                    self.face_materials,
                    jnp.where(
                        other.face_materials != -1,
                        other_face_materials_renumbered[other.face_materials],
                        other.face_materials,
                    ),
                ),
                axis=0,
            )
        elif self.face_materials is not None:
            face_materials = jnp.concatenate(
                (
                    self.face_materials,
                    jnp.full_like(self.face_materials, -1, shape=other.num_triangles),
                ),
                axis=0,
            )
        elif other.face_materials is not None:
            face_materials = jnp.concatenate(
                (
                    jnp.full_like(other.face_materials, -1, shape=self.num_triangles),
                    jnp.where(
                        other.face_materials != -1,
                        other_face_materials_renumbered[other.face_materials],
                        other.face_materials,
                    ),
                ),
                axis=0,
            )
        else:
            face_materials = None

        object_bounds = (
            jnp.concatenate(
                (self.object_bounds, other.object_bounds + other.num_triangles), axis=0
            )
            if (self.object_bounds is not None and other.object_bounds is not None)
            else None
        )
        assume_quads = self.assume_quads and other.assume_quads
        return TriangleMesh(
            vertices=vertices,
            triangles=triangles,
            face_colors=face_colors,
            face_materials=face_materials,
            object_bounds=object_bounds,
            assume_quads=assume_quads,
        )

    __add__ = append

    @overload
    def set_face_colors(
        self,
        colors: Float[ArrayLike, "#num_triangles 3"] | Float[ArrayLike, "3"],
        *,
        key: None = None,
    ) -> Self: ...

    @overload
    def set_face_colors(
        self,
        colors: None,
        *,
        key: PRNGKeyArray,
    ) -> Self: ...

    def set_face_colors(
        self,
        colors: Float[ArrayLike, "#num_triangles 3"]
        | Float[ArrayLike, "3"]
        | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Self:
        """
        Return a copy of this mesh, with new face colors.

        Args:
            colors: The array of RGB colors.
                If one color is provided, it will be applied to all triangles.

                This or ``key`` must be specified.
            key: If provided, colors will be randomly generated.

                If :attr:`object_bounds` is not :data:`None`, then triangles
                within the same object will share the same color. Otherwise,
                a random color is generated for each triangle
                (or quadrilateral if :attr:`assume_quads` is :data:`True`).

        Returns:
            A new mesh with updated face colors.

        Raises:
            ValueError: If ``colors`` or ``key`` is not specified.

        Examples:
            The following example shows how this function paints the mesh,
            for different argument types.

            First, we load a scene from Sionna :cite:`sionna`, that is
            already colored, and extract the mesh from it.

            .. plotly::
                :context:

                >>> from differt.scene import (
                ...     TriangleScene,
                ...     download_sionna_scenes,
                ...     get_sionna_scene,
                ... )
                >>>
                >>> download_sionna_scenes()  # doctest: +SKIP
                >>> file = get_sionna_scene("simple_street_canyon")
                >>> mesh = TriangleScene.load_xml(file).mesh
                >>> fig = mesh.plot(backend="plotly")
                >>> fig  # doctest: +SKIP

            Then, we could set the same color to all triangles.

            .. plotly::
                :context:

                >>> fig = mesh.set_face_colors(jnp.array([0.8, 0.2, 0.0])).plot(
                ...     backend="plotly"
                ... )
                >>> fig  # doctest: +SKIP

            We could also manually specify a different color for each triangle, but it can
            become tedious as the number of triangles gets larger. Another option is to rely
            on automatic random coloring, using the ``key`` argument.

            As our mesh is a collection of 7 distinct objects, as this was loaded from a Sionna
            XML file, this utility will automatically detect it and color each
            object differently.

            .. plotly::
                :context:

                >>> mesh.object_bounds
                Array([[ 0, 12],
                       [12, 24],
                       [24, 36],
                       [36, 48],
                       [48, 60],
                       [60, 72],
                       [72, 74]], dtype=int32)
                >>> fig = mesh.set_face_colors(key=jax.random.key(1234)).plot(
                ...     backend="plotly"
                ... )
                >>> fig  # doctest: +SKIP

            If you prefer to have per-triangle coloring, you can perform surgery on the mesh
            to remove its :attr:`object_bounds` attribute.

            .. plotly::
                :context:

                >>> import equinox as eqx
                >>>
                >>> mesh = eqx.tree_at(lambda m: m.object_bounds, mesh, None)
                >>> fig = mesh.set_face_colors(key=jax.random.key(1234)).plot(
                ...     backend="plotly"
                ... )
                >>> fig  # doctest: +SKIP

            Finally, you can also set :attr:`assume_quads` to :data:`True` to color quadrilaterals
            instead.

            .. plotly::
                :context:

                >>> fig = (
                ...     mesh.set_assume_quads()
                ...     .set_face_colors(key=jax.random.key(1234))
                ...     .plot(backend="plotly")
                ... )
                >>> fig  # doctest: +SKIP
        """
        if (colors is None) == (key is None):
            msg = "You must specify one of 'colors' or `key`, not both."
            raise ValueError(msg)

        if key is not None:
            if self.object_bounds is not None:
                object_colors = jax.random.uniform(
                    key, (self.object_bounds.shape[0], 3)
                )
                repeats = jnp.diff(self.object_bounds, axis=-1)
                colors = jnp.repeat(
                    object_colors,
                    repeats,
                    axis=0,
                    total_repeat_length=self.num_triangles,
                )
            elif self.assume_quads:
                quad_colors = jax.random.uniform(key, (self.num_quads, 3))
                repeats = jnp.full(self.num_objects, 2)
                colors = jnp.repeat(
                    quad_colors, repeats, axis=0, total_repeat_length=self.num_triangles
                )
            else:
                colors = jax.random.uniform(key, (self.num_triangles, 3))

            return self.set_face_colors(colors=colors)

        face_colors = jnp.broadcast_to(
            jnp.asarray(colors).reshape(-1, 3),
            self.triangles.shape,
        )
        return eqx.tree_at(
            lambda m: m.face_colors,
            self,
            face_colors,
            is_leaf=lambda x: x is None,
        )

    @overload
    @classmethod
    def plane(
        cls,
        vertex_a: Float[Array, "3"],
        vertex_b: Float[Array, "3"],
        vertex_c: Float[Array, "3"],
        *,
        normal: None = None,
        side_length: Float[ArrayLike, " "] = 1.0,
        rotate: Float[ArrayLike, " "] | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def plane(
        cls,
        vertex_a: Float[Array, "3"],
        vertex_b: None = None,
        vertex_c: None = None,
        *,
        normal: Float[Array, "3"],
        side_length: Float[ArrayLike, " "] = 1.0,
        rotate: Float[ArrayLike, " "] | None = None,
    ) -> Self: ...

    @classmethod
    def plane(
        cls,
        vertex_a: Float[ArrayLike, "3"],
        vertex_b: Float[ArrayLike, "3"] | None = None,
        vertex_c: Float[ArrayLike, "3"] | None = None,
        *,
        normal: Float[ArrayLike, "3"] | None = None,
        side_length: Float[ArrayLike, " "] = 1.0,
        rotate: Float[ArrayLike, " "] | None = None,
    ) -> Self:
        """
        Create a plane mesh, made of two triangles.

        Note:
            The mesh satisfies the guarantees
            expected when setting
            :attr:`assume_quads` to :data:`True`.

        Args:
            vertex_a: The center of the plane.
            vertex_b: Any second vertex on the plane.

                This and ``vertex_c``, or ``normal`` is required.
            vertex_c: Any third vertex on the plane.

                This and ``vertex_b``, or ``normal`` is required.
            normal: The plane normal.

                Must be of unit length.
            side_length: The side length of the plane.
            rotate: An optional rotation angle, in radians,
                to be applied around the normal of the plane
                and its center.

        Returns:
            A new plane mesh.

        Raises:
            ValueError: If neither ``vertex_b`` and ``vertex_c``, nor ``normal`` have been provided,
                or if both have been provided simultaneously.
        """
        if (vertex_b is None) != (vertex_c is None):
            msg = "You must specify either of both  of 'vertex_b' and 'vertex_c', or none."
            raise ValueError(msg)

        if (vertex_b is None) == (normal is None):
            msg = "You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            raise ValueError(msg)

        vertex_a = jnp.asarray(vertex_a)

        if vertex_b is not None:
            vertex_b = jnp.asarray(vertex_b)
            vertex_c = jnp.asarray(vertex_c)
            u = vertex_b - vertex_a
            v = vertex_c - vertex_a
            w = jnp.cross(u, v)
            normal = normalize(w)[0]
        else:
            normal = jnp.asarray(normal)

        u, v = orthogonal_basis(
            normal,
        )

        s = 0.5 * side_length

        vertices = s * jnp.array([u + v, v - u, -u - v, u - v])

        if rotate:
            rotation_matrix = rotation_matrix_along_axis(rotate, normal)
            vertices = (rotation_matrix @ vertices.T).T

        vertices += vertex_a

        triangles = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        return cls(vertices=vertices, triangles=triangles)

    @classmethod
    def box(
        cls,
        length: Float[ArrayLike, " "] = 1.0,
        width: Float[ArrayLike, " "] = 1.0,
        height: Float[ArrayLike, " "] = 1.0,
        *,
        with_top: bool = False,
    ) -> Self:
        """
        Create a box mesh, with an optional opening on the top.

        Note:
            The mesh satisfies the guarantees
            expected when setting
            :attr:`assume_quads` to :data:`True`.

        Args:
            length: The length of the box (along x-axis).
            width: The width of the box (along y-axis).
            height: The height of the box (along z-axis).
            with_top: Whether the top of part
                of the box is included or not.

        Returns:
            A new box mesh.

        Examples:
            The following example shows how to create a cube.

            .. plotly::

                >>> from differt.geometry import TriangleMesh
                >>> mesh = (
                ...     TriangleMesh.box(with_top=True)
                ...     .set_assume_quads()
                ...     .set_face_colors(key=jax.random.key(1234))
                ... )
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig  # doctest: +SKIP

            The second example shows how to create a corridor-like
            mesh, without the ceiling face.

            .. plotly::

                >>> from differt.geometry import TriangleMesh
                >>> mesh = (
                ...     TriangleMesh.box(length=10.0, width=3.0, height=2.0)
                ...     .set_assume_quads()
                ...     .set_face_colors(key=jax.random.key(1234))
                ... )
                >>> fig = mesh.plot(opacity=0.5, backend="plotly")
                >>> fig = fig.update_scenes(aspectmode="data")
                >>> fig  # doctest: +SKIP
        """
        dx = jnp.array([length * 0.5, 0.0, 0.0])
        dy = jnp.array([0.0, width * 0.5, 0.0])
        dz = jnp.array([0.0, 0.0, height * 0.5])

        vertices = jnp.stack((
            +dx + dy + dz,
            +dx + dy - dz,
            -dx + dy - dz,
            -dx + dy + dz,
            -dx - dy - dz,
            -dx - dy + dz,
            +dx - dy - dz,
            +dx - dy + dz,
        ))
        triangles = jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [3, 2, 4],
                [3, 4, 5],
                [5, 4, 6],
                [5, 6, 7],
                [7, 6, 1],
                [7, 1, 0],
                [1, 4, 2],  # Bottom
                [1, 6, 4],
            ],
            dtype=int,
        )
        if with_top:
            triangles = jnp.concatenate(
                (triangles, jnp.asarray([[0, 3, 5], [0, 5, 7]])),
                axis=0,
            )

        indices = jnp.arange(0, triangles.shape[0] + 1, 2)
        object_bounds = jnp.column_stack((indices[:-1], indices[+1:]))
        return cls(vertices=vertices, triangles=triangles, object_bounds=object_bounds)

    @property
    def is_empty(self) -> bool:
        """Whether this scene has no triangle."""
        return self.triangles.size == 0

    @classmethod
    def load_obj(cls, file: str) -> Self:
        """
        Load a triangle mesh from a Wavefront .obj file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        Args:
            file: The path to the Wavefront .obj file.

        Returns:
            The corresponding mesh containing only triangles.
        """
        core_mesh = differt_core.geometry.TriangleMesh.load_obj(file)
        return cls.from_core(core_mesh)

    @classmethod
    def load_ply(cls, file: str) -> Self:
        """
        Load a triangle mesh from a Stanford PLY .ply file.

        Currently, only vertices and triangles are loaded. Triangle normals
        are computed afterward (when first accessed).

        Args:
            file: The path to the Stanford PLY .ply file.

        Returns:
            The corresponding mesh containing only triangles.
        """
        core_mesh = differt_core.geometry.TriangleMesh.load_ply(file)
        return cls.from_core(core_mesh)

    def plot(self, **kwargs: Any) -> PlotOutput:
        """
        Plot this mesh on a 3D scene.

        Args:
            kwargs: Keyword arguments passed to
                :func:`draw_mesh<differt.plotting.draw_mesh>`.

        Returns:
            The resulting plot output.
        """
        if "face_colors" not in kwargs and self.face_colors is not None:
            kwargs["face_colors"] = self.face_colors

        return draw_mesh(
            vertices=self.vertices,
            triangles=self.triangles,
            **kwargs,
        )

    def sample(
        self,
        size: int,
        replace: bool = False,
        preserve: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> Self:
        """
        Generate a new mesh by randomly sampling triangles from this geometry.

        Warning:
            If :attr:`assume_quads` is :data:`True`, then quadrilaterals are
            sampled.

        Args:
            size: The size of the sample, i.e., the number of triangles.
            replace: Whether to sample with or without replacement.
            preserve: Whether to preserve :attr:`object_bounds`, otherwise
                it is discarded.

                Object bounds are re-generated by sorting the randomly generated samples,
                which takes additional time.

                Setting this to :data:`True` has no effect if :attr:`object_bounds`
                is :data:`None`.
            key: The :func:`jax.random.key` to be used.

        Returns:
            A new random mesh.
        """
        indices = jax.random.choice(
            key,
            self.num_objects,
            shape=(size,),
            replace=replace,
        )

        if self.assume_quads:
            indices *= 2

        if preserve and self.object_bounds is not None:
            indices = jnp.sort(indices)
            object_bounds = jnp.stack(
                (
                    jnp.searchsorted(indices, self.object_bounds[:, 0]),
                    jnp.searchsorted(indices, self.object_bounds[:, 1]),
                ),
                axis=-1,
            )
        else:
            object_bounds = None

        if self.assume_quads:
            indices = jnp.stack((indices, indices + 1), axis=-1).reshape(-1)

        return eqx.tree_at(
            lambda m: (
                m.vertices,
                m.triangles,
                m.face_colors,
                m.face_materials,
                m.object_bounds,
            ),
            self,
            (
                self.vertices,
                self.triangles[indices, :],
                self.face_colors[indices, :] if self.face_colors is not None else None,
                self.face_materials[indices]
                if self.face_materials is not None
                else None,
                object_bounds,
            ),
            is_leaf=lambda x: x is None,
        )

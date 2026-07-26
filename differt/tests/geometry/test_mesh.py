import logging
import re
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import pytest
from jaxtyping import Array, Float, PRNGKeyArray

from differt import rt
from differt.geometry._mesh import (
    Mesh,
    triangle_contains_vertex_assuming_inside_same_plane,
)
from differt.geometry._utils import rotation_matrix_along_x_axis

from ..utils import random_inputs


@pytest.fixture
def keep_within_mesh() -> Mesh:
    return Mesh(
        vertices=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.1, 0.0],
                [0.8, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [1.0, 0.1, 0.0],
            ],
            dtype=float,
        ),
        triangles=jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=int),
        object_bounds=jnp.array([[0, 1], [1, 3]], dtype=int),
    )


@pytest.fixture
def medium_random_mesh(key: PRNGKeyArray) -> Mesh:
    key_vertices, key_triangles = jax.random.split(key)
    vertices = jax.random.uniform(key_vertices, (64, 3), minval=-3.0, maxval=3.0)
    triangles = jax.random.randint(key_triangles, (128, 3), 0, vertices.shape[0])
    return Mesh(vertices=vertices, triangles=triangles)


@pytest.mark.parametrize(
    ("triangle_vertices", "vertices"),
    [
        ((20, 10, 3, 3), (20, 10, 3)),
        ((1, 10, 3, 3), (20, 1, 3)),
        ((10, 3, 3), (10, 3)),
        ((3, 3), (3,)),
    ],
)
@random_inputs("triangle_vertices", "vertices")
def test_triangle_contains_vertex_various_shapes(
    triangle_vertices: Array,
    vertices: Array,
) -> None:
    _ = triangle_contains_vertex_assuming_inside_same_plane(
        triangle_vertices,
        vertices,
    )


def test_triangle_contains_vertex_assuming_inside_same_planes() -> None:
    triangle_vertices = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vertices = jnp.array(
        [
            [0.1, 0.8, 0.0],  # Inside
            [0.8, 0.1, 0.0],  # Inside
            [0.1, 0.1, 0.0],  # Inside
            [0.3, 0.3, 0.0],  # Inside
            [1.0, 1.0, 0.0],  # Outside
            [0.3, 1.3, 0.0],  # Outside
            [0.0, 0.0, 0.0],  # Inside but on vertex
            [1.0, 0.0, 0.0],  # Inside but on vertex
            [0.0, 1.0, 0.0],  # Inside but on vertex
            [0.5, 0.0, 0.0],  # Inside but on edge
            [0.0, 0.5, 0.0],  # Inside but on edge
            [0.5, 0.5, 0.0],  # Inside but on edge
        ],
    )
    expected = jnp.array(
        [True, True, True, True, False, False, True, True, True, True, True, True],
    )
    n = vertices.shape[0]
    triangle_vertices = jnp.tile(triangle_vertices, (n, 1, 1))
    got = triangle_contains_vertex_assuming_inside_same_plane(
        triangle_vertices,
        vertices,
    )
    chex.assert_trees_all_equal(got, expected)


def test_triangle_mesh_deprecated() -> None:
    from differt.geometry import TriangleMesh  # ruff:ignore[import-outside-top-level]

    with pytest.warns(DeprecationWarning, match="TriangleMesh is deprecated"):
        _ = TriangleMesh(
            vertices=jnp.zeros((3, 3)), triangles=jnp.zeros((1, 3), dtype=int)
        )


class TestMesh:
    def test_init_with_non_unique_material_names(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"Material names must be unique, got \('concrete', 'glass', 'concrete'\)\.",
        ):
            _ = Mesh(
                vertices=jnp.zeros((3, 3)),
                triangles=jnp.zeros((1, 3), dtype=int),
                material_names=("concrete", "glass", "concrete"),
            )

    def test_num_triangles(self, two_buildings_mesh: Mesh) -> None:
        assert two_buildings_mesh.num_triangles == 24

    def test_num_quads(self, two_buildings_mesh: Mesh) -> None:
        with pytest.raises(
            ValueError,
            match=r"Cannot access the number of quadrilaterals if 'assume_quads' is set to 'False'.",
        ):
            _ = two_buildings_mesh.num_quads

        quad_mesh = two_buildings_mesh.set_assume_quads()

        assert quad_mesh.num_quads == 12

        non_quad_mesh = two_buildings_mesh[1:]

        with pytest.raises(
            ValueError,
            match="You cannot set 'assume_quads' to 'True' if the number of triangles is not even!",
        ):
            _ = Mesh(
                vertices=non_quad_mesh.vertices,
                triangles=non_quad_mesh.triangles,
                assume_quads=True,
            )

        _ = non_quad_mesh.set_assume_quads(flag=False)

        with pytest.raises(
            ValueError,
            match="You cannot set 'assume_quads' to 'True' if the number of triangles is not even!",
        ):
            _ = non_quad_mesh.set_assume_quads(flag=True)

        # 'tree_at' bypasses '__check_init__', so this will not raise an error
        _ = eqx.tree_at(lambda m: m.assume_quads, non_quad_mesh, replace=True)

    def test_num_primitives(self, two_buildings_mesh: Mesh) -> None:
        assert two_buildings_mesh.num_primitives == 24
        assert two_buildings_mesh.set_assume_quads().num_primitives == 12

    def test_get_item(self, two_buildings_mesh: Mesh) -> None:
        got = two_buildings_mesh[:]

        chex.assert_trees_all_equal(got, two_buildings_mesh)

        indices = jnp.arange(two_buildings_mesh.num_triangles)

        got = two_buildings_mesh[indices]

        chex.assert_trees_all_equal(got, two_buildings_mesh)

        got = two_buildings_mesh[::2]

        assert got.num_triangles == two_buildings_mesh.num_triangles // 2

        # TODO: test that other attributes are set correctly.

    def test_iter_objects(self) -> None:
        mesh = Mesh.empty()
        assert mesh.num_triangles == 0

        count = 0
        for sub_mesh in mesh.iter_objects():
            count += 1
            assert sub_mesh.num_triangles == 0

        assert count == 1

        mesh = Mesh.box(with_top=True)
        assert mesh.num_triangles == 12

        count = 0
        for sub_mesh in mesh.iter_objects():
            count += 1
            assert sub_mesh.num_triangles == 2

            sub_count = 0
            for sub_sub_mesh in sub_mesh.iter_objects():
                sub_count += 1
                assert sub_sub_mesh.num_triangles == 2

            assert sub_count == 1

        assert count == 6

        mesh = eqx.tree_at(lambda m: m.object_bounds, mesh, None)

        count = 0
        for sub_mesh in mesh.iter_objects():
            count += 1
            assert sub_mesh.num_triangles == 12

        assert count == 1

    def test_keep_any_within(self, keep_within_mesh: Mesh) -> None:
        mesh = keep_within_mesh

        filtered = mesh.keep_any_within(x_min=0.75, preserve_objects=False)
        chex.assert_trees_all_equal(
            filtered.mask, jnp.array([False, False, True], dtype=bool)
        )
        assert filtered.num_active_triangles == 1
        assert filtered.masked().num_triangles == 1

        assert (
            mesh
            .keep_all_within(x_min=0.75, preserve_objects=False)
            .masked()
            .num_triangles
            == 1
        )

        preserved = mesh.keep_any_within(x_min=0.75, preserve_objects=True)
        chex.assert_trees_all_equal(
            preserved.mask, jnp.array([False, True, True], dtype=bool)
        )
        assert preserved.num_active_triangles == 2
        assert preserved.masked().num_triangles == 2

    def test_keep_all_within(self, keep_within_mesh: Mesh) -> None:
        mesh = keep_within_mesh

        filtered = mesh.keep_all_within(x_min=0.75, preserve_objects=False)
        chex.assert_trees_all_equal(
            filtered.mask, jnp.array([False, False, True], dtype=bool)
        )
        assert filtered.num_active_triangles == 1
        assert filtered.masked().num_triangles == 1

        preserved = mesh.keep_all_within(x_min=0.75, preserve_objects=True)
        chex.assert_trees_all_equal(
            preserved.mask, jnp.array([False, False, False], dtype=bool)
        )
        assert preserved.num_active_triangles == 0
        assert preserved.masked().num_triangles == 0

    def test_keep_within_respects_existing_mask(self, keep_within_mesh: Mesh) -> None:
        mesh = eqx.tree_at(
            lambda m: m.mask,
            keep_within_mesh,
            jnp.array([False, False, True], dtype=bool),
            is_leaf=lambda x: x is None,
        )

        preserved = mesh.keep_all_within(x_min=0.75, preserve_objects=True)

        chex.assert_trees_all_equal(
            preserved.mask, jnp.array([False, False, True], dtype=bool)
        )
        assert preserved.num_active_triangles == 1
        assert preserved.masked().num_triangles == 1

    def test_keep_within_preserve_objects_complex(self) -> None:
        # Create a mesh with 3 objects:
        # Object 0: triangles [0, 1]
        # Object 1: triangles [2, 3, 4]
        # Object 2: triangles [5, 6]
        vertices = jnp.array(
            [
                # Object 0, Triangle 0: x in [1.0, 2.0]
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.5, 1.0, 0.0],
                # Object 0, Triangle 1: x in [-1.0, 0.0]
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.5, 1.0, 0.0],
                # Object 1, Triangle 2: x in [1.5, 2.5]
                [1.5, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                # Object 1, Triangle 3: x in [2.0, 3.0]
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.5, 1.0, 0.0],
                # Object 1, Triangle 4: x in [-2.0, -1.0]
                [-2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.5, 1.0, 0.0],
                # Object 2, Triangle 5: x in [1.0, 2.0]
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.5, 1.0, 0.0],
                # Object 2, Triangle 6: x in [0.2, 0.5]
                [0.2, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.3, 1.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.arange(21, dtype=int).reshape(7, 3)
        object_bounds = jnp.array([[0, 2], [2, 5], [5, 7]], dtype=int)

        mesh = Mesh(
            vertices=vertices,
            triangles=triangles,
            object_bounds=object_bounds,
        )

        # Set an initial mask
        mesh = eqx.tree_at(
            lambda m: m.mask,
            mesh,
            jnp.array([True, False, True, True, False, True, True], dtype=bool),
            is_leaf=lambda x: x is None,
        )

        preserved_all = mesh.keep_all_within(x_min=1.0, preserve_objects=True)

        chex.assert_trees_all_equal(
            preserved_all.mask,
            jnp.array([True, False, True, True, False, False, False], dtype=bool),
        )

        preserved_any = mesh.keep_any_within(x_min=1.0, preserve_objects=True)

        chex.assert_trees_all_equal(
            preserved_any.mask,
            jnp.array([True, False, True, True, False, True, True], dtype=bool),
        )

    @pytest.mark.parametrize("method_name", ["keep_all_within", "keep_any_within"])
    def test_keep_within_preserve_objects_without_bounds(
        self, method_name: str
    ) -> None:
        mesh = Mesh(
            vertices=jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.8, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [1.0, 0.1, 0.0],
                ],
                dtype=float,
            ),
            triangles=jnp.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        )

        filtered = getattr(mesh, method_name)(x_min=0.75, preserve_objects=False)
        preserved = getattr(mesh, method_name)(x_min=0.75, preserve_objects=True)

        chex.assert_trees_all_equal(filtered.mask, preserved.mask)
        chex.assert_trees_all_equal(filtered.masked(), preserved.masked())

    def test_clip(self) -> None:
        mesh = Mesh(
            vertices=jnp.array(
                [[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [7.0, 8.0, -9.0]],
                dtype=float,
            ),
            triangles=jnp.array([[0, 1, 2]], dtype=int),
        )

        got = mesh.clip(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0)
        expected = Mesh(
            vertices=jnp.array(
                [[0.0, 1.0, 3.0], [1.0, 0.0, 6.0], [1.0, 1.0, 0.0]],
                dtype=float,
            ),
            triangles=jnp.array([[0, 1, 2]], dtype=int),
        )

        chex.assert_trees_all_equal(got, expected)

    @pytest.mark.parametrize("x_min", [None, -1.0])
    @pytest.mark.parametrize("x_max", [None, +1.0])
    @pytest.mark.parametrize("y_min", [None, -0.5])
    @pytest.mark.parametrize("y_max", [None, +0.5])
    @pytest.mark.parametrize("z_min", [None, -2.0])
    @pytest.mark.parametrize("z_max", [None, +2.0])
    def test_clip_random_medium_mesh(
        self,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
        z_min: float | None,
        z_max: float | None,
        medium_random_mesh: Mesh,
    ) -> None:
        clipped = medium_random_mesh.clip(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

        assert clipped.triangles.shape == medium_random_mesh.triangles.shape

        if x_min is not None:
            assert jnp.all(clipped.vertices[:, 0] >= x_min)
        if x_max is not None:
            assert jnp.all(clipped.vertices[:, 0] <= x_max)
        if y_min is not None:
            assert jnp.all(clipped.vertices[:, 1] >= y_min)
        if y_max is not None:
            assert jnp.all(clipped.vertices[:, 1] <= y_max)
        if z_min is not None:
            assert jnp.all(clipped.vertices[:, 2] >= z_min)
        if z_max is not None:
            assert jnp.all(clipped.vertices[:, 2] <= z_max)

        if all(bound is None for bound in (x_min, x_max, y_min, y_max, z_min, z_max)):
            chex.assert_trees_all_equal(clipped.vertices, medium_random_mesh.vertices)

    @pytest.mark.parametrize("method_name", ["keep_all_within", "keep_any_within"])
    @pytest.mark.parametrize("x_min", [None, -1.0])
    @pytest.mark.parametrize("x_max", [None, +1.0])
    @pytest.mark.parametrize("y_min", [None, -0.5])
    @pytest.mark.parametrize("y_max", [None, +0.5])
    @pytest.mark.parametrize("z_min", [None, -2.0])
    @pytest.mark.parametrize("z_max", [None, +2.0])
    def test_keep_within_random_medium_mesh(
        self,
        method_name: str,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
        z_min: float | None,
        z_max: float | None,
        medium_random_mesh: Mesh,
    ) -> None:
        kept = getattr(medium_random_mesh, method_name)(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            preserve_objects=False,
            clip=True,
        )

        assert kept.triangles.shape == medium_random_mesh.triangles.shape
        assert kept.mask is not None

        if x_min is not None:
            assert jnp.all(kept.vertices[:, 0] >= x_min)
        if x_max is not None:
            assert jnp.all(kept.vertices[:, 0] <= x_max)
        if y_min is not None:
            assert jnp.all(kept.vertices[:, 1] >= y_min)
        if y_max is not None:
            assert jnp.all(kept.vertices[:, 1] <= y_max)
        if z_min is not None:
            assert jnp.all(kept.vertices[:, 2] >= z_min)
        if z_max is not None:
            assert jnp.all(kept.vertices[:, 2] <= z_max)

        if all(bound is None for bound in (x_min, x_max, y_min, y_max, z_min, z_max)):
            chex.assert_trees_all_equal(kept.vertices, medium_random_mesh.vertices)
            chex.assert_trees_all_equal(
                kept.mask, jnp.ones(medium_random_mesh.num_triangles, dtype=bool)
            )

    def test_keep_within_clip(self, keep_within_mesh: Mesh) -> None:
        expected_keep_all_vertices = jnp.array(
            [
                [0.75, 0.0, 0.0],
                [0.75, 0.0, 0.0],
                [0.75, 0.2, 0.0],
                [0.75, 0.0, 0.0],
                [0.75, 0.0, 0.0],
                [0.75, 0.1, 0.0],
                [0.8, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [1.0, 0.1, 0.0],
            ],
            dtype=float,
        )
        expected_keep_any_vertices = expected_keep_all_vertices

        mesh = keep_within_mesh

        kept = mesh.keep_all_within(
            x_min=0.75,
            preserve_objects=True,
            clip=True,
        )
        chex.assert_trees_all_equal(
            kept.mask, jnp.array([False, False, False], dtype=bool)
        )
        chex.assert_trees_all_equal(kept.vertices, expected_keep_all_vertices)

        kept = keep_within_mesh.keep_any_within(
            x_min=0.75,
            preserve_objects=False,
            clip=True,
        )
        chex.assert_trees_all_equal(
            kept.mask, jnp.array([False, False, True], dtype=bool)
        )
        chex.assert_trees_all_equal(kept.vertices, expected_keep_any_vertices)

    @pytest.mark.xfail(
        reason="No longer raises an error, as no more type checking is done on jnp.asarray.",
    )
    def test_invalid_args(self) -> None:
        vertices = jnp.ones((10, 2), dtype=float)
        triangles = jnp.ones((20, 3), dtype=int)

        with pytest.raises(jaxtyping.TypeCheckError):
            _ = Mesh(vertices=vertices, triangles=triangles)

        vertices = jnp.ones((10, 3), dtype=float)
        triangles = jnp.ones((20, 3), dtype=float)

        with pytest.raises(jaxtyping.TypeCheckError):
            _ = Mesh(vertices=vertices, triangles=triangles)

    def test_plane(self, key: PRNGKeyArray) -> None:
        center = jnp.ones(3, dtype=float)
        normal = jnp.array([0.0, 0.0, 1.0])
        mesh = Mesh.plane(center, normal=normal, side_length=2.0)

        got = mesh.bounding_box
        expected = jnp.array([[0.0, 0.0, 1.0], [2.0, 2.0, 1.0]])

        chex.assert_trees_all_equal(got, expected)

        rotated_mesh = Mesh.plane(
            center, normal=normal, side_length=2.0, rotate=jnp.pi / 4
        )

        got = rotated_mesh.bounding_box
        inc = jnp.sqrt(2) - 1
        expected = jnp.array([[0.0 - inc, 0.0 - inc, 1.0], [2.0 + inc, 2.0 + inc, 1.0]])

        chex.assert_trees_all_equal(got, expected)

        vertex_a, vertex_b, vertex_c = jax.random.uniform(key, (3, 3)).T
        _ = Mesh.plane(vertex_a, vertex_b, vertex_c)

        with pytest.raises(
            ValueError,
            match=r"You must specify either of both  of 'vertex_b' and 'vertex_c', or none.",
        ):
            _ = Mesh.plane(vertex_a, vertex_b)  # type: ignore[ty:no-matching-overload]

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify either of both  of 'vertex_b' and 'vertex_c', or none."
            ),
        ):
            _ = Mesh.plane(vertex_a, vertex_c=vertex_c)  # type: ignore[ty:no-matching-overload]

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            ),
        ):
            _ = Mesh.plane(vertex_a, vertex_b, vertex_c, normal=normal)

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            ),
        ):
            _ = Mesh.plane(vertex_a, vertex_b, vertex_c, normal=normal)

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            ),
        ):
            _ = Mesh.plane(center)  # type: ignore[ty:no-matching-overload]

    @pytest.mark.parametrize(
        ("length", "width", "height"),
        [(10.0, 5.0, 4.0)],
    )
    @pytest.mark.parametrize("with_top", [False, True])
    @pytest.mark.parametrize("with_bottom", [False, True])
    def test_box(
        self,
        length: float,
        width: float,
        height: float,
        with_top: bool,
        with_bottom: bool,
    ) -> None:
        mesh = Mesh.box(
            length, width, height, with_top=with_top, with_bottom=with_bottom
        )

        if with_top and with_bottom:
            assert mesh.num_triangles == 12
        elif with_top or with_bottom:
            assert mesh.num_triangles == 10
        else:
            assert mesh.num_triangles == 8

        dx = length * 0.5
        dy = width * 0.5
        dz = height * 0.5

        assert mesh.bounding_box.tolist() == [[-dx, -dy, -dz], [+dx, +dy, +dz]]

    @pytest.mark.parametrize(
        "index",
        [
            slice(None),
            slice(0, None, 2),
            jnp.arange(24),
            jnp.array([0, 1, 2]),
            jnp.ones(24, dtype=bool),
            jnp.array([0, 3, 3, 4, 4, 5]),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "jax_method", "func_or_values"),
        [
            ("get", "get", ()),
            ("set", "set", (0.0,)),
            ("add", "add", (jnp.array([1.0, 3.0, 6.0]),)),
            ("sub", "subtract", (jnp.array([1.0, 3.0, 6.0]),)),
            ("mul", "mul", (2.0,)),
            ("div", "divide", (2.0,)),
            ("pow", "power", (2.0,)),
            ("min", "min", (jnp.array([0.0, 0.0, 0.0]),)),
            ("max", "max", (jnp.array([100.0, 100.0, 100.0]),)),
            ("apply", "apply", (lambda x: 1 / x,)),
        ],
    )
    def test_at_update(
        self,
        index: slice | Array,
        method: str,
        jax_method: str,
        func_or_values: tuple[Any, ...],
        two_buildings_mesh: Mesh,
    ) -> None:
        got = getattr(two_buildings_mesh.at[index], method)(*func_or_values)

        if method != "get" and isinstance(index, Array) and index.dtype != jnp.bool:
            # This should be a no-op, because duplicate indices are dropped before updating
            index = jnp.unique(index)
        index = two_buildings_mesh.triangles[index, :].reshape(-1)
        if method != "get":
            # Duplicate indices are dropped before updating
            index = jnp.unique(index)

        vertices = getattr(two_buildings_mesh.vertices.at[index, :], jax_method)(
            *func_or_values
        )
        if method == "get":
            expected = vertices
        else:
            expected = eqx.tree_at(
                lambda m: m.vertices,
                two_buildings_mesh,
                vertices,
            )
        chex.assert_trees_all_equal(got, expected)

    @pytest.mark.require_no_typechecker
    def test_at_update_invalid_index_value_error(
        self, two_buildings_mesh: Mesh
    ) -> None:
        with pytest.raises(ValueError, match="Index must be at most one-dimensional"):
            two_buildings_mesh.at[jnp.ones((2, 3), dtype=int)]

    def test_rotate(self, two_buildings_mesh: Mesh, key: PRNGKeyArray) -> None:
        angle = jax.random.uniform(key, (), minval=0, maxval=2 * jnp.pi)

        got = two_buildings_mesh.rotate(rotation_matrix_along_x_axis(angle)).rotate(
            rotation_matrix_along_x_axis(-angle)
        )
        chex.assert_trees_all_close(got, two_buildings_mesh, atol=1e-5)

        got = two_buildings_mesh.rotate(rotation_matrix_along_x_axis(angle)).rotate(
            rotation_matrix_along_x_axis(2 * jnp.pi - angle)
        )
        chex.assert_trees_all_close(got, two_buildings_mesh, atol=1e-4)

        got = two_buildings_mesh.rotate(rotation_matrix_along_x_axis(0.0))
        chex.assert_trees_all_close(got, two_buildings_mesh)

    def test_scale(self, two_buildings_mesh: Mesh, key: PRNGKeyArray) -> None:
        scale_factor = jax.random.uniform(key, (), minval=1.5, maxval=2)

        got = two_buildings_mesh.scale(scale_factor).scale(1 / scale_factor)
        chex.assert_trees_all_close(got, two_buildings_mesh)

        got = two_buildings_mesh.scale(1.0)
        chex.assert_trees_all_close(got, two_buildings_mesh)

    def test_translate(self, two_buildings_mesh: Mesh, key: PRNGKeyArray) -> None:
        translation = jax.random.normal(key, (3,))

        got = two_buildings_mesh.translate(translation).translate(-translation)
        chex.assert_trees_all_close(got, two_buildings_mesh)

        got = two_buildings_mesh.translate(jnp.zeros_like(translation))
        chex.assert_trees_all_close(got, two_buildings_mesh)

    def test_empty(self) -> None:
        assert Mesh.empty().is_empty

    def test_not_empty(self, two_buildings_mesh: Mesh) -> None:
        assert not two_buildings_mesh.is_empty

    @pytest.mark.parametrize(
        "self_empty",
        [False, True],
    )
    @pytest.mark.parametrize(
        "other_empty",
        [False, True],
    )
    @pytest.mark.parametrize(
        "self_assume_quads",
        [False, True],
    )
    @pytest.mark.parametrize(
        "other_assume_quads",
        [False, True],
    )
    @pytest.mark.parametrize(
        "self_colors",
        [False, True],
    )
    @pytest.mark.parametrize(
        "other_colors",
        [False, True],
    )
    @pytest.mark.parametrize(
        "self_face_materials",
        [False, True],
    )
    @pytest.mark.parametrize(
        "other_face_materials",
        [False, True],
    )
    @pytest.mark.parametrize(
        "self_mask",
        [False, True],
    )
    @pytest.mark.parametrize(
        "other_mask",
        [False, True],
    )
    def test_append(
        self,
        self_empty: bool,
        other_empty: bool,
        self_assume_quads: bool,
        other_assume_quads: bool,
        self_colors: bool,
        other_colors: bool,
        self_face_materials: bool,
        other_face_materials: bool,
        self_mask: bool,
        other_mask: bool,
        two_buildings_mesh: Mesh,
        key: PRNGKeyArray,
    ) -> None:
        s = (Mesh.empty() if self_empty else two_buildings_mesh).set_assume_quads(
            self_assume_quads
        )
        o = (Mesh.empty() if other_empty else two_buildings_mesh).set_assume_quads(
            other_assume_quads
        )

        key_s, key_o = jax.random.split(key)

        if self_colors and not self_empty:
            s = s.set_face_colors(key=key_s)
        if other_colors and not other_empty:
            o = o.set_face_colors(key=key_o)

        if self_face_materials and not self_empty:
            s = s.set_materials("material_a")
        if other_face_materials and not other_empty:
            o = o.set_materials("material_b")

        if self_mask and not self_empty:
            s = eqx.tree_at(
                lambda m: m.mask,
                s,
                jnp.ones(s.triangles.shape[0], dtype=bool),
                is_leaf=lambda x: x is None,
            )

        if other_mask and not other_empty:
            o = eqx.tree_at(
                lambda m: m.mask,
                o,
                jnp.ones(o.triangles.shape[0], dtype=bool),
                is_leaf=lambda x: x is None,
            )

        mesh = s + o

        assert mesh.num_triangles == (s.num_triangles + o.num_triangles)

        if (
            (self_assume_quads and not self_empty)
            and (other_empty or other_assume_quads)
        ) or (
            (other_assume_quads and not other_empty)
            and (self_empty or self_assume_quads)
        ):
            assert mesh.num_primitives == mesh.num_quads
        else:
            assert mesh.num_primitives == mesh.num_triangles

        if (self_colors and not self_empty) or (other_colors and not other_empty):
            assert mesh.face_colors is not None
        else:
            assert mesh.face_colors is None

        # TODO: Test merging material names and indices.

        if (self_mask and not self_empty) or (other_mask and not other_empty):
            assert mesh.mask is not None
        else:
            assert mesh.mask is None

        chex.assert_trees_all_equal(mesh, s.append(o))

    def test_drop_duplicates(self, key: PRNGKeyArray) -> None:
        def sort_mesh(mesh: Mesh) -> Mesh:
            indices = jnp.lexsort(mesh.vertices.T[::-1])
            vertices = mesh.vertices[indices]
            triangles = jnp.argsort(indices)[mesh.triangles]
            return eqx.tree_at(
                lambda m: (m.vertices, m.triangles),
                mesh,
                (vertices, triangles),
            )

        mesh = Mesh.box()
        got_mesh = sum(mesh.iter_objects(), start=Mesh.empty()).drop_duplicates()
        expected_mesh = sort_mesh(mesh)

        chex.assert_trees_all_equal(sort_mesh(got_mesh), expected_mesh)

        mesh_dup = Mesh.box().drop_duplicates()
        got_mesh_sampled = sort_mesh(
            mesh_dup.sample(size=mesh_dup.num_triangles, preserve=True, key=key)
        )
        chex.assert_trees_all_equal(got_mesh_sampled, sort_mesh(mesh_dup))

    @pytest.mark.parametrize(
        "shape",
        [
            (3,),
            (1, 3),
            (24, 3),
        ],
    )
    def test_set_face_colors(
        self,
        shape: tuple[int, ...],
        two_buildings_mesh: Mesh,
        key: PRNGKeyArray,
    ) -> None:
        colors = jax.random.uniform(key, shape)
        assert two_buildings_mesh.face_colors is None
        mesh = two_buildings_mesh.set_face_colors(colors)
        assert mesh.face_colors is not None

    def test_set_face_colors_wrong_args(
        self,
        two_buildings_mesh: Mesh,
        key: PRNGKeyArray,
    ) -> None:
        colors = jax.random.uniform(key, (two_buildings_mesh.num_triangles, 3))
        with pytest.raises(
            ValueError, match="You must specify one of 'colors' or `key`, not both"
        ):
            _ = two_buildings_mesh.set_face_colors(colors, key=key)  # type: ignore[ty:no-matching-overload]

    def test_set_face_materials(
        self,
    ) -> None:
        mesh = Mesh.box(with_top=True)
        assert mesh.face_materials is None
        assert len(mesh.material_names) == 0

        mesh = mesh.set_materials("concrete")
        assert mesh.face_materials is not None
        assert len(mesh.material_names) == 1
        chex.assert_trees_all_equal(mesh.face_materials, jnp.zeros(12, dtype=int))
        mesh = mesh.set_materials(*["glass"] * 12)
        chex.assert_trees_all_equal(mesh.face_materials, jnp.ones(12, dtype=int))
        assert len(mesh.material_names) == 2
        with pytest.raises(
            ValueError,
            match="Expected either 1, or 12 names, got 2",
        ):
            _ = mesh.set_materials("glass", "brick")
        mesh = mesh.set_assume_quads()
        mesh = mesh.set_materials("metal")
        with pytest.raises(
            ValueError,
            match="Expected either 1, 12, or 6 names, got 4",
        ):
            _ = mesh.set_materials("glass", "brick", "wood", "plastic")
        mesh = mesh.set_materials(
            "metal", "glass", "concrete", "brick", "wood", "plastic"
        )
        assert mesh.material_names == (
            "concrete",
            "glass",
            "metal",
            "brick",
            "wood",
            "plastic",
        )
        mesh = mesh.set_materials(
            "metal", "glass", "concrete", "brick", "wood", "plastic"
        )
        mesh = mesh.set_materials("concrete")

    def test_load_obj(self, two_buildings_obj_file: str) -> None:
        mesh = Mesh.load_obj(two_buildings_obj_file)
        assert mesh.triangles.shape == (24, 3)

    def test_load_obj_with_mat(self, two_buildings_obj_with_mat_file: str) -> None:
        mesh = Mesh.load_obj(two_buildings_obj_with_mat_file)
        assert mesh.triangles.shape == (24, 3)
        assert len(mesh.material_names) == 2
        assert {material_name.lower() for material_name in mesh.material_names} == {
            "concrete",
            "glass",
        }
        assert mesh.face_colors is not None
        assert mesh.face_materials is not None

    def test_load_ply(self, two_buildings_ply_file: str) -> None:
        mesh = Mesh.load_ply(two_buildings_ply_file)
        assert mesh.triangles.shape == (24, 3)

    def test_load_ply_with_colors(
        self, cube_ply_file: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.clear()

        with caplog.at_level(logging.INFO):
            mesh = Mesh.load_ply(cube_ply_file)

            assert mesh.triangles.shape == (2, 3)
            assert (
                len([
                    record
                    for record in caplog.records
                    if "because it is not a triangle" in record.msg
                ])
                == 5
            )

    def test_compare_with_open3d(
        self,
        two_buildings_obj_file: str,
        two_buildings_mesh: Mesh,
    ) -> None:
        o3d = pytest.importorskip("open3d", reason="open3d not installed")
        mesh = o3d.io.read_triangle_mesh(
            two_buildings_obj_file,
        ).compute_triangle_normals()

        got_triangles = two_buildings_mesh.triangles
        expected_triangles = jnp.asarray(mesh.triangles, dtype=int)

        got_vertices = two_buildings_mesh.vertices
        expected_vertices = jnp.asarray(mesh.vertices)

        got_all_vertices = jnp.take(got_vertices, got_triangles, axis=0)
        expected_all_vertices = jnp.take(expected_vertices, expected_triangles, axis=0)

        chex.assert_trees_all_close(got_all_vertices, expected_all_vertices)

        got_normals = two_buildings_mesh.normals
        expected_normals = jnp.asarray(mesh.triangle_normals)

        chex.assert_trees_all_close(
            got_normals,
            expected_normals,
            atol=1e-7,
        )

    def test_normals(self, two_buildings_mesh: Mesh) -> None:
        chex.assert_equal_shape(
            (two_buildings_mesh.normals, two_buildings_mesh.triangles),
        )
        got = jnp.linalg.norm(two_buildings_mesh.normals, axis=-1)
        expected = jnp.ones_like(got)
        chex.assert_trees_all_close(got, expected)

    def test_plot(self, sphere_mesh: Mesh) -> None:
        sphere_mesh.plot()

    def test_sample(self, two_buildings_mesh: Mesh, key: PRNGKeyArray) -> None:
        assert two_buildings_mesh.sample(10, key=key).num_triangles == 10

        with pytest.raises(ValueError, match="Cannot take a larger sample"):
            two_buildings_mesh.sample(30, key=key)

        assert two_buildings_mesh.sample(30, replace=True, key=key).num_triangles == 30

        assert two_buildings_mesh.set_assume_quads().sample(5, key=key).num_quads == 5

        with pytest.raises(
            ValueError,
            match="Cannot sample by objects when 'object_bounds' is None",
        ):
            two_buildings_mesh.sample(0.5, sample_objects=True, key=key)

        two_buildings_mesh = eqx.tree_at(
            lambda m: m.object_bounds,
            two_buildings_mesh,
            jnp.array([[0, 12], [12, 24]]),
            is_leaf=lambda x: x is None,
        )

        assert two_buildings_mesh.sample(
            13, key=key, preserve=True
        ).object_bounds.shape == (2, 2)

        assert two_buildings_mesh.sample(
            2, key=key, preserve=True, sample_objects=True
        ).object_bounds.shape == (2, 2)

        assert two_buildings_mesh.sample(
            1, key=key, preserve=True, sample_objects=True
        ).object_bounds.shape == (1, 2)

        assert two_buildings_mesh.sample(
            1, key=key, preserve=True, by_masking=True, sample_objects=True
        ).object_bounds.shape == (2, 2)

        with pytest.raises(
            TypeError, match="'size' must be an integer when 'by_masking' is False"
        ):
            two_buildings_mesh.sample(0.5, key=key)

        with pytest.raises(
            ValueError,
            match="Cannot sample with replacement when 'by_masking' is True",
        ):
            eqx.filter_jit(two_buildings_mesh.sample)(
                10, replace=True, by_masking=True, key=key
            )

        with pytest.raises(
            ValueError,
            match="Cannot preserve 'object_bounds' when 'by_masking' is True and 'sample_objects' is False",
        ):
            eqx.filter_jit(two_buildings_mesh.sample)(
                0.5, preserve=True, by_masking=True, key=key
            )

        assert (
            eqx.filter_jit(two_buildings_mesh.sample)(
                0.5, preserve=True, by_masking=True, sample_objects=True, key=key
            )
            is not None
        )
        assert (
            eqx.filter_jit(two_buildings_mesh.sample)(
                0.5, preserve=False, by_masking=True, sample_objects=True, key=key
            )
            is not None
        )

        assert two_buildings_mesh.mask is None
        # Sampling creates a mask when by_masking=True
        assert (
            eqx.filter_jit(two_buildings_mesh.sample)(
                0.5, by_masking=True, key=key
            ).mask
            is not None
        )
        # Sampling works with quads when by_masking=True
        assert (
            two_buildings_mesh
            .set_assume_quads()
            .sample(0.5, by_masking=True, key=key)
            .mask
            is not None
        )
        # Sampling preserves mask when by_masking=False
        assert (
            two_buildings_mesh
            .sample(10, by_masking=True, key=key)
            .sample(10, key=key)
            .mask
            is not None
        )

        assert (
            two_buildings_mesh
            .sample(5, by_masking=True, key=key)
            .masked()
            .num_triangles
            == 5
        )
        assert (
            two_buildings_mesh
            .set_assume_quads()
            .sample(5, by_masking=True, key=key)
            .masked()
            .num_triangles
            == 10
        )

    @pytest.mark.parametrize(
        ("with_colors", "with_materials", "with_mask", "assume_quads"),
        [
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (True, True, False, False),
            (True, False, True, False),
            (False, True, True, False),
            (True, True, True, False),
            (False, False, False, True),
            (True, False, False, True),
        ],
    )
    def test_shuffle_with_configurations(
        self,
        two_buildings_mesh: Mesh,
        key: PRNGKeyArray,
        with_colors: bool,
        with_materials: bool,
        with_mask: bool,
        assume_quads: bool,
    ) -> None:
        # Test shuffling with various mesh configurations
        key_color, key_shuffle = jax.random.split(key)
        mesh = two_buildings_mesh

        if assume_quads:
            mesh = mesh.set_assume_quads()

        if with_colors:
            mesh = mesh.set_face_colors(key=key_color)

        if with_materials:
            mesh = mesh.set_materials("concrete")

        if with_mask:
            mesh = eqx.tree_at(
                lambda m: m.mask,
                mesh,
                jnp.ones(mesh.num_triangles, dtype=bool),
                is_leaf=lambda x: x is None,
            )

        shuffled_mesh, indices = mesh.shuffle(return_indices=True, key=key_shuffle)

        # Verify mesh structure is preserved
        assert shuffled_mesh.num_triangles == mesh.num_triangles
        assert shuffled_mesh.assume_quads == mesh.assume_quads

        # Indices should be a permutation of triangle indices
        assert indices.shape == (mesh.num_triangles,)
        assert jnp.all(jnp.unique(indices) == jnp.arange(mesh.num_triangles))

        # Triangles should be reordered by indices
        expected_triangles = mesh.triangles[indices, :]
        chex.assert_trees_all_equal(shuffled_mesh.triangles, expected_triangles)

        # Face colors should be reordered if present
        if mesh.face_colors is not None:
            expected_colors = mesh.face_colors[indices, :]
            chex.assert_trees_all_close(shuffled_mesh.face_colors, expected_colors)

        # Face materials should be reordered if present
        if mesh.face_materials is not None:
            expected_materials = mesh.face_materials[indices]
            chex.assert_trees_all_equal(
                shuffled_mesh.face_materials, expected_materials
            )

        # Mask should be reordered if present
        if mesh.mask is not None:
            expected_mask = mesh.mask[indices]
            chex.assert_trees_all_equal(shuffled_mesh.mask, expected_mask)

    def test_shuffle_basic(self, two_buildings_mesh: Mesh, key: PRNGKeyArray) -> None:
        # Test basic shuffling of triangles
        shuffled_mesh = two_buildings_mesh.shuffle(key=key)

        # Mesh should have same number of triangles
        assert shuffled_mesh.num_triangles == two_buildings_mesh.num_triangles

        # Vertices and triangles should be different after shuffling (with high probability)
        assert not jnp.array_equal(
            shuffled_mesh.triangles, two_buildings_mesh.triangles
        )

        # All vertices should still be present
        chex.assert_equal_shape(
            (shuffled_mesh.vertices, two_buildings_mesh.vertices),
        )

    def test_shuffle_empty_mesh(self, key: PRNGKeyArray) -> None:
        # Test shuffling an empty mesh
        empty_mesh = Mesh.empty()
        shuffled_mesh = empty_mesh.shuffle(key=key)

        assert shuffled_mesh.is_empty
        assert shuffled_mesh.num_triangles == 0

    @pytest.mark.parametrize(
        ("has_object_bounds", "preserve"),
        [
            (True, False),
            (True, True),
            (False, True),
        ],
    )
    def test_shuffle_object_bounds(
        self,
        two_buildings_mesh: Mesh,
        key: PRNGKeyArray,
        has_object_bounds: bool,
        preserve: bool,
    ) -> None:
        # Test shuffling with object_bounds handling
        if has_object_bounds:
            mesh = eqx.tree_at(
                lambda m: m.object_bounds,
                two_buildings_mesh,
                jnp.array([[0, 12], [12, 24]]),
                is_leaf=lambda x: x is None,
            )
        else:
            mesh = two_buildings_mesh

        if preserve:
            # preserve=True always raises NotImplementedError
            with pytest.raises(NotImplementedError, match="Preserving object bounds"):
                mesh.shuffle(preserve=True, key=key)
        else:
            # Without preserve, object_bounds should be removed
            shuffled_mesh = mesh.shuffle(key=key)
            assert shuffled_mesh.object_bounds is None

    def test_center_simple_mesh(self) -> None:
        # Test centering a simple mesh
        # Create a mesh with known bounds
        vertices = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2], [1, 3, 2]], dtype=int)
        mesh = Mesh(vertices=vertices, triangles=triangles)

        # Center the mesh
        centered = mesh.center()

        # The center of the bounding box should be at (1.0, 1.0, z_min)
        # After centering, it should be at (0.0, 0.0, z_min)
        (x_min, y_min, z_min), (x_max, y_max, z_max) = centered.bounding_box
        assert jnp.allclose(x_min + x_max, 0.0, atol=1e-6)
        assert jnp.allclose(y_min + y_max, 0.0, atol=1e-6)

        # The z-coordinate should remain unchanged
        assert jnp.allclose(z_min, 0.0, atol=1e-6)
        assert jnp.allclose(z_max, 0.0, atol=1e-6)

    def test_center_asymmetric_mesh(self) -> None:
        # Test centering a mesh with asymmetric bounds
        vertices = jnp.array(
            [
                [1.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [1.0, 4.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2], [1, 3, 2]], dtype=int)
        mesh = Mesh(vertices=vertices, triangles=triangles)

        centered = mesh.center()

        # Center should be at (0, 0) after centering
        (x_min, y_min, _), (x_max, y_max, _) = centered.bounding_box
        assert jnp.allclose(x_min + x_max, 0.0, atol=1e-6)
        assert jnp.allclose(y_min + y_max, 0.0, atol=1e-6)

    def test_center_with_negative_z(self) -> None:
        # Test centering a mesh with negative z values
        vertices = jnp.array(
            [
                [0.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [2.0, 2.0, 1.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2], [1, 3, 2]], dtype=int)
        mesh = Mesh(vertices=vertices, triangles=triangles)

        centered = mesh.center()

        # Z-coordinates should not be modified
        assert jnp.allclose(jnp.min(centered.vertices[:, 2]), -1.0, atol=1e-6)
        assert jnp.allclose(jnp.max(centered.vertices[:, 2]), 1.0, atol=1e-6)

        # X and Y should be centered
        (x_min, y_min, _), (x_max, y_max, _) = centered.bounding_box
        assert jnp.allclose(x_min + x_max, 0.0, atol=1e-6)
        assert jnp.allclose(y_min + y_max, 0.0, atol=1e-6)

    def test_center_preserves_structure(self) -> None:
        # Test that centering preserves mesh structure and attributes
        vertices = jnp.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2]], dtype=int)
        face_colors = jnp.array([[1.0, 0.0, 0.0]], dtype=float)
        mesh = Mesh(
            vertices=vertices,
            triangles=triangles,
            face_colors=face_colors,
        )

        centered = mesh.center()

        # Mesh structure should be preserved
        assert centered.num_triangles == mesh.num_triangles
        chex.assert_trees_all_equal(centered.triangles, mesh.triangles)
        chex.assert_trees_all_equal(centered.face_colors, mesh.face_colors)

    def test_center_idempotent(self) -> None:
        # Test that centering twice gives the same result as centering once
        vertices = jnp.array(
            [
                [1.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [1.0, 4.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2], [1, 3, 2]], dtype=int)
        mesh = Mesh(vertices=vertices, triangles=triangles)

        centered_once = mesh.center()
        centered_twice = centered_once.center()

        # Centering twice should not change the result (within tolerance)
        chex.assert_trees_all_close(
            centered_once.vertices,
            centered_twice.vertices,
            atol=1e-6,
        )

    def test_center_with_mask(self) -> None:
        # Test centering a masked mesh
        vertices = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array(
            [
                [0, 1, 2],
                [1, 3, 2],
                [1, 4, 5],
            ],
            dtype=int,
        )
        mask = jnp.array([True, True, False], dtype=bool)
        mesh = Mesh(
            vertices=vertices,
            triangles=triangles,
            mask=mask,
        )

        centered = mesh.center()

        # Mask should be preserved
        chex.assert_trees_all_equal(centered.mask, mesh.mask)

    def test_center_single_point_mesh(self) -> None:
        # Test centering a degenerate mesh with all vertices at the same point
        vertices = jnp.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2]], dtype=int)
        mesh = Mesh(vertices=vertices, triangles=triangles)

        centered = mesh.center()

        # All vertices should be at the origin in x,y
        assert jnp.allclose(centered.vertices[:, 0], 0.0, atol=1e-6)
        assert jnp.allclose(centered.vertices[:, 1], 0.0, atol=1e-6)

    def test_center_with_object_bounds(self) -> None:
        # Test centering a mesh with object bounds
        vertices = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [3.0, 2.0, 0.0],
                [5.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array(
            [
                [0, 1, 2],
                [1, 3, 2],
                [4, 5, 6],
                [5, 7, 6],
            ],
            dtype=int,
        )
        object_bounds = jnp.array([[0, 2], [2, 4]], dtype=int)
        mesh = Mesh(
            vertices=vertices,
            triangles=triangles,
            object_bounds=object_bounds,
        )

        centered = mesh.center()

        # Object bounds should be preserved
        chex.assert_trees_all_equal(centered.object_bounds, mesh.object_bounds)

    def test_add_ground_simple(self) -> None:
        # Test adding a ground plane to a simple mesh
        mesh = Mesh(
            vertices=jnp.array(
                [
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [0.0, 2.0, 1.0],
                    [2.0, 2.0, 1.0],
                ],
                dtype=float,
            ),
            triangles=jnp.array([[0, 1, 2], [1, 3, 2]], dtype=jnp.int32),
        )

        with_ground = mesh.add_ground()

        # The mesh should have 2 additional triangles (1 quadrilateral = 2 triangles)
        assert with_ground.num_triangles == mesh.num_triangles + 2

        # The ground should be below the mesh
        mesh_min_z = jnp.min(mesh.vertices[:, 2])
        ground_max_z = jnp.max(with_ground.vertices[int(mesh.vertices.shape[0]) :, 2])
        assert jnp.allclose(ground_max_z, mesh_min_z, atol=1e-6)

    def test_add_ground_has_correct_structure(self) -> None:
        # Test that the added ground plane has the correct structure
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0)
        with_ground = mesh.add_ground()

        # The ground plane should be represented as a quadrilateral (2 triangles)
        ground_triangles = with_ground.triangles[int(mesh.triangles.shape[0]) :, :]
        assert int(ground_triangles.shape[0]) == 2
        assert int(ground_triangles.shape[1]) == 3

        # The last 4 vertices should form a quadrilateral
        ground_vertices = with_ground.vertices[int(mesh.vertices.shape[0]) :, :]
        assert int(ground_vertices.shape[0]) == 4
        assert int(ground_vertices.shape[1]) == 3

    def test_add_ground_centered(self) -> None:
        # Test that the ground plane is centered on the mesh
        # Create a simple mesh without using translate to avoid type errors
        mesh = Mesh(
            vertices=jnp.array(
                [
                    [1.0, 2.0, 1.0],
                    [3.0, 2.0, 1.0],
                    [1.0, 4.0, 1.0],
                    [3.0, 4.0, 1.0],
                ],
                dtype=float,
            ),
            triangles=jnp.array([[0, 1, 2], [1, 3, 2]], dtype=jnp.int32),
        )
        with_ground = mesh.add_ground()

        # The mesh center should match the ground center (in x, y)
        mesh_center_x = (
            jnp.min(mesh.vertices[:, 0]) + jnp.max(mesh.vertices[:, 0])
        ) / 2
        mesh_center_y = (
            jnp.min(mesh.vertices[:, 1]) + jnp.max(mesh.vertices[:, 1])
        ) / 2

        ground_vertices = with_ground.vertices[mesh.vertices.shape[0] :, :]
        ground_center_x = (
            jnp.min(ground_vertices[:, 0]) + jnp.max(ground_vertices[:, 0])
        ) / 2
        ground_center_y = (
            jnp.min(ground_vertices[:, 1]) + jnp.max(ground_vertices[:, 1])
        ) / 2

        assert jnp.allclose(mesh_center_x, ground_center_x, atol=1e-6)
        assert jnp.allclose(mesh_center_y, ground_center_y, atol=1e-6)

    def test_add_ground_custom_z(self) -> None:
        # Test adding a ground plane at a custom z coordinate
        # Create a mesh at a higher altitude
        mesh = Mesh(
            vertices=jnp.array(
                [
                    [0.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=float,
            ),
            triangles=jnp.array([[0, 1, 2], [1, 3, 2]], dtype=jnp.int32),
        )
        custom_z = -5.0
        with_ground = mesh.add_ground(z=custom_z)

        # All ground vertices should be at custom_z
        ground_vertices = with_ground.vertices[int(mesh.vertices.shape[0]) :, :]
        assert jnp.allclose(ground_vertices[:, 2], custom_z, atol=1e-6)

    def test_add_ground_custom_scale(self) -> None:
        # Test adding a ground plane with custom scaling
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0)
        with_ground_default = mesh.add_ground(x_scale=1.0, y_scale=1.0)
        with_ground_scaled = mesh.add_ground(x_scale=2.0, y_scale=2.0)

        # The scaled ground should be larger
        ground_default = with_ground_default.vertices[int(mesh.vertices.shape[0]) :, :]
        ground_scaled = with_ground_scaled.vertices[int(mesh.vertices.shape[0]) :, :]

        default_width = jnp.max(ground_default[:, 0]) - jnp.min(ground_default[:, 0])
        scaled_width = jnp.max(ground_scaled[:, 0]) - jnp.min(ground_scaled[:, 0])
        assert scaled_width > default_width

    def test_add_ground_custom_color(self) -> None:
        # Test adding a ground plane with custom color
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0)
        custom_color = jnp.array([0.0, 1.0, 0.0], dtype=float)  # Green
        with_ground = mesh.add_ground(face_color=custom_color)

        # The ground face colors should be set to custom color
        face_colors = with_ground.face_colors
        assert face_colors is not None
        ground_colors = face_colors[int(mesh.triangles.shape[0]) :, :]
        chex.assert_trees_all_close(
            ground_colors, jnp.array([custom_color, custom_color], dtype=float)
        )

    def test_add_ground_custom_material(self) -> None:
        # Test adding a ground plane with custom material name
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0)
        custom_material = "custom_ground"
        with_ground = mesh.add_ground(material_name=custom_material)

        # The material name should be included
        assert custom_material in with_ground.material_names

    def test_add_ground_is_quad(self) -> None:
        # Test that the ground plane itself is a valid quadrilateral
        # Create a mesh that has assume_quads set
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0).set_assume_quads()
        with_ground = mesh.add_ground()

        # The resulting mesh should have assume_quads set since both original mesh and ground have it
        assert with_ground.assume_quads is True
        # The total number of triangles should be even (compatible with quads)
        assert (with_ground.num_triangles % 2) == 0

    def test_add_ground_preserves_original_mesh(self) -> None:
        # Test that the original mesh is preserved when adding ground
        original_vertices = jnp.array(
            [
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2], [1, 3, 2]], dtype=jnp.int32)
        mesh = Mesh(vertices=original_vertices, triangles=triangles)

        with_ground = mesh.add_ground()

        # The original mesh vertices should be the same
        chex.assert_trees_all_close(
            with_ground.vertices[: original_vertices.shape[0], :],
            original_vertices,
            atol=1e-6,
        )

    def test_add_ground_object_bounds(self) -> None:
        # Test that object bounds are set for the ground plane
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0)
        with_ground = mesh.add_ground()

        # Object bounds should be set for the ground
        assert with_ground.object_bounds is not None
        # The added object bounds should reference the ground triangles
        # which are at the end
        assert int(with_ground.object_bounds.shape[0]) >= 1

    def test_add_ground_chaining(self) -> None:
        # Test that add_ground can be chained with other methods
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0)
        result = mesh.center().add_ground()

        assert result.num_triangles == mesh.num_triangles + 2

    def test_add_ground_small_mesh(self) -> None:
        # Test adding ground to a small degenerate mesh
        vertices = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [0.0, 0.001, 0.0],
            ],
            dtype=float,
        )
        triangles = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        mesh = Mesh(vertices=vertices, triangles=triangles)

        with_ground = mesh.add_ground()

        # Should still add ground without issues
        assert with_ground.num_triangles == 3

    def test_add_ground_multiple_times(self) -> None:
        # Test adding ground multiple times
        mesh = Mesh.box(length=2.0, width=2.0, height=1.0)
        with_ground_once = mesh.add_ground()
        with_ground_twice = with_ground_once.add_ground()

        # Each call adds 2 triangles
        assert with_ground_once.num_triangles == mesh.num_triangles + 2
        assert with_ground_twice.num_triangles == mesh.num_triangles + 4

    def test_add_ground_after_filtering(self) -> None:
        # Test adding ground after filtering operations
        mesh = Mesh.box(length=4.0, width=2.0, height=1.0)
        # Filter and then add ground
        filtered = mesh.keep_all_within(x_max=2.0, preserve_objects=False)
        with_ground = filtered.masked().add_ground()

        # Should successfully add ground to filtered mesh
        assert (
            with_ground.num_triangles > mesh.num_triangles
        )  # At least 2 triangles for ground


class TestMeshDiffraction:
    def test_boundary_edges(self) -> None:
        # A single triangle has 3 boundary edges, 0 diffraction edges
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        triangles = jnp.array([[0, 1, 2]])
        mesh = Mesh(vertices=vertices, triangles=triangles)

        assert mesh.diffraction_edges.shape[0] == 0

    def test_diffraction_edges_isolated(self) -> None:
        # A single triangle, but we mock diffraction_edges_mask to return True for one edge
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        triangles = jnp.array([[0, 1, 2]])

        class MockMesh(Mesh):
            @property
            def diffraction_edges_mask(self) -> Array:
                return jnp.array([[True, False, False]])

        mesh = MockMesh(
            vertices=vertices, triangles=triangles, assume_unique_vertices=True
        )

        # There should be 1 unique diffraction edge
        assert mesh.diffraction_edges.shape[0] == 1

        # Its adjacent triangles should be [[0, -1]] because it's isolated!
        adj_t = mesh.diffraction_edges_to_triangles
        assert adj_t.shape == (1, 2)
        chex.assert_trees_all_close(adj_t, jnp.array([[0, -1]]))

    def test_coplanar_edges(self) -> None:
        # Two coplanar triangles forming a quad
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        # Triangle 1: 0, 1, 2. Triangle 2: 1, 3, 2.
        # Shared edge between 1 and 2 is [1, 2] which is sorted (1, 2)
        triangles = jnp.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        mesh = Mesh(vertices=vertices, triangles=triangles, assume_quads=False)

        # They share an edge but they are coplanar, so they should NOT be diffraction edges.
        assert not jnp.any(mesh.diffraction_edges_mask)

    def test_convex_and_concave_wedges(self) -> None:
        # A right-angle convex wedge (like a building corner)
        # Triangle 1 on the top face (z = 0, normal [0, 0, 1])
        # Triangle 2 on the side face (x = 1, normal [1, 0, 0])
        vertices = jnp.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [1.0, 0.0, -1.0],  # 3
        ])
        # Triangle 1: (0, 1, 2) -> normal [0, 0, 1]
        # Triangle 2: (1, 3, 2) -> normal [1, 0, 0]
        triangles = jnp.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        mesh = Mesh(vertices=vertices, triangles=triangles, assume_quads=False)

        # Edge (1, 2) is shared.
        # Normal 1 = [0, 0, 1], Normal 2 = [1, 0, 0].
        # Cos angle is 0, so angle phi is pi/2.
        # The corner is convex, so exterior angle is 1.5 * pi. n = 1.5.
        mask = mesh.diffraction_edges_mask
        assert (
            jnp.sum(mask) == 2
        )  # shared edge is represented once for each of the two adjacent triangles

        n = mesh.wedge_parameters
        assert n.shape == (1,)
        chex.assert_trees_all_close(n, jnp.array([1.5]))

        # Check adjacent triangles matrix
        adj_t = mesh.diffraction_edges_to_triangles
        assert adj_t.shape == (1, 2)
        assert set(adj_t[0].tolist()) == {0, 1}

    def test_assume_quads(self) -> None:
        # Two coplanar triangles forming a quad
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        triangles = jnp.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        mesh = Mesh(vertices=vertices, triangles=triangles, assume_quads=True)

        # The shared diagonal should be ignored because assume_quads=True.
        adj_t, _ = mesh._connectivity()  # ruff:ignore[private-member-access]
        assert jnp.all(adj_t == -1)

    def test_non_manifold_edges(self) -> None:
        # Three triangles sharing a single edge
        vertices = jnp.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [0.0, -1.0, 0.0],  # 4
        ])
        triangles = jnp.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ])

        mesh = Mesh(vertices=vertices, triangles=triangles)
        with pytest.warns(UserWarning, match="The mesh contains non-manifold edges"):
            _ = mesh.diffraction_edges_mask

    def test_duplicate_vertices(self) -> None:
        box = Mesh.box(with_top=True, with_bottom=True)
        mesh = Mesh.empty()
        for plane in box.iter_objects():
            mesh += plane

        assert not mesh.assume_unique_vertices

        # Check that diffraction edges are correctly identified
        edges = mesh.diffraction_edges
        assert edges.shape[0] == 12

        # Check adjacent triangles matrix
        adj_t = mesh.diffraction_edges_to_triangles
        assert adj_t.shape == (12, 2)

        # We can also check that dedup_vertices itself works as expected.
        deduped = mesh.dedup_vertices()
        assert deduped.assume_unique_vertices
        assert deduped.vertices.shape[0] == mesh.vertices.shape[0]

        # And after dropping unused vertices, the number of vertices is reduced.
        dropped = deduped.drop_unused_vertices()
        assert dropped.vertices.shape[0] < mesh.vertices.shape[0]

    def test_dedup_vertices_coverage(self) -> None:
        # 1. No-op if assume_unique_vertices is already True
        mesh_unique = Mesh(
            vertices=jnp.array([[0.0, 0.0, 0.0]]),
            triangles=jnp.array([[0, 0, 0]]),
            assume_unique_vertices=True,
        )
        chex.assert_trees_all_equal(mesh_unique.dedup_vertices(), mesh_unique)

        # 2. Empty mesh with len(vertices) == 0 and assume_unique_vertices=False
        mesh_empty = Mesh(
            vertices=jnp.empty((0, 3)),
            triangles=jnp.empty((0, 3), dtype=int),
            assume_unique_vertices=False,
        )
        dedup_empty = mesh_empty.dedup_vertices()
        assert dedup_empty.assume_unique_vertices
        assert dedup_empty.vertices.shape[0] == 0

        # 3. Rounding effect check
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [0.0001, 0.0, 0.0],
        ])
        triangles = jnp.array([
            [0, 0, 1],
        ])
        mesh = Mesh(vertices=vertices, triangles=triangles)

        # Without decimals, they are not merged
        dedup_no_decimals = mesh.dedup_vertices()
        chex.assert_trees_all_equal(dedup_no_decimals.triangles, triangles)

        # With num_decimals=2, they round to the same coordinates and are merged
        dedup_decimals = mesh.dedup_vertices(num_decimals=2)
        expected_triangles = jnp.array([
            [0, 0, 0],
        ])
        chex.assert_trees_all_equal(dedup_decimals.triangles, expected_triangles)

    def test_empty_mesh_diffraction_properties(self) -> None:
        # Create an empty mesh with assume_unique_vertices=True
        mesh = Mesh.empty()

        # This will call _connectivity() with num_triangles == 0
        adj_t, adj_e = mesh._connectivity()  # ruff:ignore[private-member-access]
        assert adj_t.shape == (0, 3)
        assert adj_e.shape == (0, 3)

        # This will call diffraction_edges_mask with num_triangles == 0
        mask = mesh.diffraction_edges_mask
        assert mask.shape == (0, 3)

        # This will call wedge_angles with num_triangles == 0
        angles = mesh.wedge_angles
        assert angles.shape == (0, 3)

    def test_wedge_angles_non_unique_vertices(self) -> None:
        # A box split into planes (so duplicate vertices and assume_unique_vertices=False)
        box = Mesh.box()
        mesh = Mesh.empty()
        for plane in box.iter_objects():
            mesh += plane

        assert not mesh.assume_unique_vertices

        # This will call wedge_angles when assume_unique_vertices is False
        angles = mesh.wedge_angles
        # Since it dedups first, it should run successfully and return wedge angles
        assert angles.shape == (mesh.dedup_vertices().num_triangles, 3)

    def test_diffraction_edges_with_mask(self) -> None:
        # Two adjacent triangles sharing an edge (not coplanar)
        vertices = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        triangles = jnp.array([
            [0, 1, 2],
            [0, 3, 1],
        ])
        mesh = Mesh(vertices=vertices, triangles=triangles, assume_unique_vertices=True)

        # By default, both triangles are active, and edge (0, 1) is a diffraction edge.
        assert mesh.diffraction_edges.shape[0] == 1

        # Now apply a mask where only triangle 0 is active (triangle 1 is inactive)
        masked_mesh = eqx.tree_at(
            lambda m: m.mask,
            mesh,
            jnp.array([True, False]),
            is_leaf=lambda x: x is None,
        )

        # Since triangle 1 is inactive, the shared edge (0, 1) is no longer a valid diffraction edge
        assert masked_mesh.diffraction_edges.shape[0] == 0

    def test_drop_unused_vertices_empty(self) -> None:
        mesh = Mesh(
            vertices=jnp.empty((0, 3)),
            triangles=jnp.empty((0, 3), dtype=int),
        )
        assert mesh.drop_unused_vertices().vertices.shape[0] == 0

    def test_ray_intersect_any_triangle_correctness(self, key: PRNGKeyArray) -> None:
        mesh = Mesh.box(2.0, 2.0, 2.0)

        key_origins, key_directions = jax.random.split(key)
        ray_origins = jax.random.uniform(key_origins, (10, 3), minval=-5.0, maxval=5.0)
        ray_directions = jax.random.uniform(
            key_directions, (10, 3), minval=-1.0, maxval=1.0
        )

        expected = rt.ray_intersect_any_triangle(
            ray_origins,
            ray_directions,
            mesh.triangle_vertices,
        )
        got = mesh.ray_intersect_any_triangle(
            ray_origins,
            ray_directions,
        )
        chex.assert_trees_all_equal(got, expected)

    def test_first_triangle_hit_by_ray_correctness_and_gradients(
        self, key: PRNGKeyArray
    ) -> None:
        mesh = Mesh.box(2.0, 2.0, 2.0)

        key_origins, key_directions = jax.random.split(key)
        ray_origins_rand = jax.random.uniform(
            key_origins, (10, 3), minval=-5.0, maxval=5.0
        )
        ray_directions_rand = jax.random.uniform(
            key_directions, (10, 3), minval=-1.0, maxval=1.0
        )

        expected_idx, expected_t = rt.first_triangle_hit_by_ray(
            ray_origins_rand,
            ray_directions_rand,
            mesh.triangle_vertices,
        )
        got_idx, got_t = mesh.first_triangle_hit_by_ray(
            ray_origins_rand,
            ray_directions_rand,
        )
        chex.assert_trees_all_equal(got_idx, expected_idx)
        chex.assert_trees_all_close(got_t, expected_t, rtol=1e-5, atol=1e-5)

        ray_origins = jnp.array([
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        ray_directions = jnp.array([
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])

        def ref_fun(
            origins: Float[Array, "num_rays 3"],
            directions: Float[Array, "num_rays 3"],
            vertices: Float[Array, "num_vertices 3"],
        ) -> Float[Array, " num_rays"]:
            triangle_vertices = jnp.take(vertices, mesh.triangles, axis=0)
            _, t = rt.first_triangle_hit_by_ray(
                origins,
                directions,
                triangle_vertices,
            )
            return t

        def got_fun(
            origins: Float[Array, "num_rays 3"],
            directions: Float[Array, "num_rays 3"],
            vertices: Float[Array, "num_vertices 3"],
        ) -> Float[Array, " num_rays"]:
            m = eqx.tree_at(lambda x: x.vertices, mesh, vertices)
            _, t = m.first_triangle_hit_by_ray(
                origins,
                directions,
            )
            return t

        jac_ref = jax.jacobian(ref_fun, argnums=(0, 1, 2))(
            ray_origins, ray_directions, mesh.vertices
        )
        jac_got = jax.jacobian(got_fun, argnums=(0, 1, 2))(
            ray_origins, ray_directions, mesh.vertices
        )

        for j_got, j_ref in zip(jac_got, jac_ref, strict=True):
            chex.assert_trees_all_close(j_got, j_ref, rtol=1e-5, atol=1e-5)

    def test_triangles_visible_from_vertex_correctness(self, key: PRNGKeyArray) -> None:
        mesh = Mesh.box(2.0, 2.0, 2.0)

        key_origins, _ = jax.random.split(key)
        ray_origins = jax.random.uniform(key_origins, (10, 3), minval=-5.0, maxval=5.0)

        expected = rt.triangles_visible_from_vertex(
            ray_origins,
            mesh.triangle_vertices,
            num_rays=100,
        )
        got = mesh.triangles_visible_from_vertex(
            ray_origins,
            num_rays=100,
        )
        chex.assert_trees_all_equal(got, expected)

import logging
import re
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import pytest
from jaxtyping import Array, PRNGKeyArray

from differt.geometry._triangle_mesh import (
    TriangleMesh,
    triangles_contain_vertices_assuming_inside_same_plane,
)
from differt.geometry._utils import rotation_matrix_along_x_axis

from ..utils import random_inputs


@pytest.mark.parametrize(
    ("triangle_vertices", "vertices", "expectation"),
    [
        ((20, 10, 3, 3), (20, 10, 3), does_not_raise()),
        ((1, 10, 3, 3), (20, 1, 3), does_not_raise()),
        ((10, 3, 3), (10, 3), does_not_raise()),
        ((3, 3), (3,), does_not_raise()),
        (
            (3, 3),
            (4,),
            pytest.raises(TypeError),
        ),
        (
            (10, 3, 3),
            (12, 3),
            pytest.raises(TypeError),
        ),
    ],
)
@random_inputs("triangle_vertices", "vertices")
def test_triangles_contain_vertices_assuming_inside_same_planes_random_inputs(
    triangle_vertices: Array,
    vertices: Array,
    expectation: AbstractContextManager[Exception],
) -> None:
    with expectation:
        _ = triangles_contain_vertices_assuming_inside_same_plane(
            triangle_vertices,
            vertices,
        )


def test_triangles_contain_vertices_assuming_inside_same_planes() -> None:
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
    got = triangles_contain_vertices_assuming_inside_same_plane(
        triangle_vertices,
        vertices,
    )
    chex.assert_trees_all_equal(got, expected)


class TestTriangleMesh:
    def test_num_triangles(self, two_buildings_mesh: TriangleMesh) -> None:
        assert two_buildings_mesh.num_triangles == 24

    def test_num_quads(self, two_buildings_mesh: TriangleMesh) -> None:
        with pytest.raises(
            ValueError,
            match=r"Cannot access the number of quadrilaterals if 'assume_quads' is set to 'False'.",
        ):
            _ = two_buildings_mesh.num_quads

        quad_mesh = eqx.tree_at(
            lambda m: m.assume_quads, two_buildings_mesh, replace=True
        )

        assert quad_mesh.num_quads == 12

        non_quad_mesh = two_buildings_mesh[1:]

        with pytest.raises(
            ValueError,
            match="You cannot set 'assume_quads' to 'True' if the number of triangles is not even!",
        ):
            _ = TriangleMesh(
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

    def test_num_objects(self, two_buildings_mesh: TriangleMesh) -> None:
        assert two_buildings_mesh.num_objects == 24
        assert two_buildings_mesh.set_assume_quads().num_objects == 12

    def test_get_item(self, two_buildings_mesh: TriangleMesh) -> None:
        got = two_buildings_mesh[:]

        chex.assert_trees_all_equal(got, two_buildings_mesh)

        indices = jnp.arange(two_buildings_mesh.num_triangles)

        got = two_buildings_mesh[indices]

        chex.assert_trees_all_equal(got, two_buildings_mesh)

        got = two_buildings_mesh[::2]

        assert got.num_triangles == two_buildings_mesh.num_triangles // 2

        # TODO: test that other attributes are set correctly.

    def test_iter_objects(self) -> None:
        mesh = TriangleMesh.empty()
        assert mesh.num_triangles == 0

        count = 0
        for sub_mesh in mesh.iter_objects():
            count += 1
            assert sub_mesh.num_triangles == 0

        assert count == 1

        mesh = TriangleMesh.box(with_top=True)
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

    def test_invalid_args(self) -> None:
        vertices = jnp.ones((10, 2))
        triangles = jnp.ones((20, 3))

        with pytest.raises(jaxtyping.TypeCheckError):
            _ = TriangleMesh(vertices=vertices, triangles=triangles)

    def test_plane(self, key: PRNGKeyArray) -> None:
        center = jnp.ones(3, dtype=float)
        normal = jnp.array([0.0, 0.0, 1.0])
        mesh = TriangleMesh.plane(center, normal=normal, side_length=2.0)

        got = mesh.bounding_box
        expected = jnp.array([[0.0, 0.0, 1.0], [2.0, 2.0, 1.0]])

        chex.assert_trees_all_equal(got, expected)

        rotated_mesh = TriangleMesh.plane(
            center, normal=normal, side_length=2.0, rotate=jnp.pi / 4
        )

        got = rotated_mesh.bounding_box
        inc = jnp.sqrt(2) - 1
        expected = jnp.array([[0.0 - inc, 0.0 - inc, 1.0], [2.0 + inc, 2.0 + inc, 1.0]])

        chex.assert_trees_all_equal(got, expected)

        vertex_a, vertex_b, vertex_c = jax.random.uniform(key, (3, 3)).T
        _ = TriangleMesh.plane(vertex_a, vertex_b, vertex_c)

        with pytest.raises(
            ValueError,
            match=r"You must specify either of both  of 'vertex_b' and 'vertex_c', or none.",
        ):
            _ = TriangleMesh.plane(vertex_a, vertex_b)  # type: ignore[reportCallIssue]

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify either of both  of 'vertex_b' and 'vertex_c', or none."
            ),
        ):
            _ = TriangleMesh.plane(vertex_a, vertex_c=vertex_c)  # type: ignore[reportCallIssue]

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            ),
        ):
            _ = TriangleMesh.plane(vertex_a, vertex_b, vertex_c, normal=normal)

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            ),
        ):
            _ = TriangleMesh.plane(vertex_a, vertex_b, vertex_c, normal=normal)

        with pytest.raises(
            ValueError,
            match=re.escape(
                r"You must specify one of ('vertex_b', 'vertex_c') or 'normal', not both."
            ),
        ):
            _ = TriangleMesh.plane(center)  # type: ignore[reportCallIssue]

    @pytest.mark.parametrize(
        ("length", "width", "height"),
        [(10.0, 5.0, 4.0)],
    )
    @pytest.mark.parametrize("with_top", [False, True])
    def test_box(
        self, length: float, width: float, height: float, with_top: bool
    ) -> None:
        mesh = TriangleMesh.box(length, width, height, with_top=with_top)

        if with_top:
            assert mesh.num_triangles == 12
        else:
            assert mesh.num_triangles == 10

        dx = length * 0.5
        dy = width * 0.5
        dz = height * 0.5

        assert mesh.bounding_box.tolist() == [[-dx, -dy, -dz], [+dx, +dy, +dz]]

    def test_rotate(self, two_buildings_mesh: TriangleMesh, key: PRNGKeyArray) -> None:
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

    def test_scale(self, two_buildings_mesh: TriangleMesh, key: PRNGKeyArray) -> None:
        scale_factor = jax.random.uniform(key, (), minval=1.5, maxval=2)

        got = two_buildings_mesh.scale(scale_factor).scale(1 / scale_factor)
        chex.assert_trees_all_close(got, two_buildings_mesh)

        got = two_buildings_mesh.scale(1.0)
        chex.assert_trees_all_close(got, two_buildings_mesh)

    def test_translate(
        self, two_buildings_mesh: TriangleMesh, key: PRNGKeyArray
    ) -> None:
        translation = jax.random.normal(key, (3,))

        got = two_buildings_mesh.translate(translation).translate(-translation)
        chex.assert_trees_all_close(got, two_buildings_mesh)

        got = two_buildings_mesh.translate(jnp.zeros_like(translation))
        chex.assert_trees_all_close(got, two_buildings_mesh)

    def test_empty(self) -> None:
        assert TriangleMesh.empty().is_empty

    def test_not_empty(self, two_buildings_mesh: TriangleMesh) -> None:
        assert not two_buildings_mesh.is_empty

    @pytest.mark.parametrize(
        ("shape", "expectation"),
        [
            ((3,), does_not_raise()),
            ((1, 3), does_not_raise()),
            ((24, 3), does_not_raise()),
            pytest.param(
                (30, 3),
                pytest.raises(TypeError),
                marks=pytest.mark.xfail(
                    reason="Unsupported type checking of typing.Self"
                ),
            ),
            pytest.param(
                (1, 24, 3),
                pytest.raises(TypeError),
                marks=pytest.mark.xfail(
                    reason="Unsupported type checking of typing.Self"
                ),
            ),
        ],
    )
    def test_set_face_colors(
        self,
        shape: tuple[int, ...],
        expectation: AbstractContextManager[Exception],
        two_buildings_mesh: TriangleMesh,
        key: PRNGKeyArray,
    ) -> None:
        colors = jax.random.uniform(key, shape)
        assert two_buildings_mesh.face_colors is None
        with expectation:
            mesh = two_buildings_mesh.set_face_colors(colors)
            assert mesh.face_colors is not None

    def test_set_face_colors_wrong_args(
        self,
        two_buildings_mesh: TriangleMesh,
        key: PRNGKeyArray,
    ) -> None:
        colors = jax.random.uniform(key, (two_buildings_mesh.num_triangles, 3))
        with pytest.raises(
            ValueError, match="You must specify one of 'colors' or `key`, not both"
        ):
            _ = two_buildings_mesh.set_face_colors(colors, key=key)  # type: ignore[reportCallIssue]

    def test_load_obj(self, two_buildings_obj_file: str) -> None:
        mesh = TriangleMesh.load_obj(two_buildings_obj_file)
        assert mesh.triangles.shape == (24, 3)

    def test_load_obj_with_mat(self, two_buildings_obj_with_mat_file: str) -> None:
        mesh = TriangleMesh.load_obj(two_buildings_obj_with_mat_file)
        assert mesh.triangles.shape == (24, 3)
        assert len(mesh.material_names) == 2
        assert {material_name.lower() for material_name in mesh.material_names} == {
            "concrete",
            "glass",
        }
        assert mesh.face_colors is not None
        assert mesh.face_materials is not None

    def test_load_ply(self, two_buildings_ply_file: str) -> None:
        mesh = TriangleMesh.load_ply(two_buildings_ply_file)
        assert mesh.triangles.shape == (24, 3)

    def test_load_ply_with_colors(
        self, cube_ply_file: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.clear()

        with caplog.at_level(logging.INFO):
            mesh = TriangleMesh.load_ply(cube_ply_file)

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
        two_buildings_mesh: TriangleMesh,
    ) -> None:
        o3d = pytest.importorskip("open3d")
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

    def test_normals(self, two_buildings_mesh: TriangleMesh) -> None:
        chex.assert_equal_shape(
            (two_buildings_mesh.normals, two_buildings_mesh.triangles),
        )
        got = jnp.linalg.norm(two_buildings_mesh.normals, axis=-1)
        expected = jnp.ones_like(got)
        chex.assert_trees_all_close(got, expected)

    def test_plot(self, sphere_mesh: TriangleMesh) -> None:
        sphere_mesh.plot()

    def test_sample(self, two_buildings_mesh: TriangleMesh, key: PRNGKeyArray) -> None:
        assert two_buildings_mesh.sample(10, key=key).num_triangles == 10

        with pytest.raises(ValueError, match="Cannot take a larger sample"):
            assert two_buildings_mesh.sample(30, key=key)

        assert two_buildings_mesh.sample(30, replace=True, key=key).num_triangles == 30

        assert two_buildings_mesh.set_assume_quads().sample(5, key=key).num_quads == 5

        two_buildings_mesh = eqx.tree_at(
            lambda m: m.object_bounds,
            two_buildings_mesh,
            jnp.array([[0, 12], [12, 24]]),
            is_leaf=lambda x: x is None,
        )

        assert two_buildings_mesh.sample(
            13, key=key, preserve=True
        ).object_bounds.shape == (2, 2)  # type: ignore[reportOptionalMemberAccess]

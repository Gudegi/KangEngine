"""Validate App-side MCP authoring with native prims, without a GL context."""

from types import SimpleNamespace

import numpy as np
import pytest

import kangengine as ke
from kangengine.app.mcp.authoring import SceneAuthoring
from kangengine.app.capture import AutomationError


@pytest.fixture
def fixture():
    root = ke.scene.Prim("", ke.scene.PrimType.ROOT)
    parent = root.add_child("group", ke.scene.PrimType.XFORM)
    parent.set_local_translation([10, 0, 0])
    prim = parent.add_child("box", ke.scene.PrimType.MESH)
    app = SimpleNamespace(
        scene=SimpleNamespace(
            get_root_prim=lambda: root, get_prim_at_path=root.get_prim_at_path
        ),
        _existing_resource_handle=lambda material: None,
    )
    return app, root, prim, SceneAuthoring()


def test_complete_local_trs_preserves_parent_and_normalizes_quaternion(fixture):
    app, root, prim, editor = fixture
    result = editor.scene_set_transform(
        app,
        path="/group/box",
        translation=[1, 2, 3],
        rotation_wxyz=[2, 0, 0, 0],
        scale=[2, 3, 4],
    )
    np.testing.assert_allclose(np.asarray(result["world_matrix"])[:3, 3], [11, 2, 3])
    np.testing.assert_allclose(np.diag(result["local_matrix"]), [2, 3, 4, 1])
    assert not result["automation_created"]
    editor.scene_set_transform(app, path="/group/box", translation=[0, 0, 0])
    np.testing.assert_allclose(
        np.asarray(prim.compute_local_matrix()), np.diag([2, 3, 4, 1])
    )


@pytest.mark.parametrize("magnitude", [1e-7, 1.0, 1e5])
def test_authored_trs_matches_native_transform(fixture, magnitude):
    app, root, prim, editor = fixture
    rotation = np.array([2, -3, 4, 5], dtype=float)
    expected = root.add_child("expected", ke.scene.PrimType.XFORM)
    expected.set_local_translation([1, 2, 3])
    expected.set_local_rotation((rotation / np.linalg.norm(rotation)).tolist())
    expected.set_local_scale([2, 3, 4])

    editor.scene_set_transform(
        app,
        path="/group/box",
        translation=[1, 2, 3],
        rotation_wxyz=rotation * magnitude,
        scale=[2, 3, 4],
    )
    np.testing.assert_allclose(
        prim.compute_local_matrix(), expected.compute_local_matrix(), atol=2e-6
    )


def test_partial_trs_preserves_omitted_fields(fixture):
    app, root, prim, editor = fixture
    editor.scene_set_transform(
        app,
        path="/group/box",
        translation=[1, 2, 3],
        rotation_wxyz=[1, 0, 0, 1],
        scale=[2, 3, 4],
    )
    before = np.asarray(prim.compute_local_matrix()).copy()
    editor.scene_set_transform(app, path="/group/box", scale=[4, 6, 8])
    after = np.asarray(prim.compute_local_matrix()).copy()
    np.testing.assert_allclose(after[:3, :3], before[:3, :3] * 2)
    np.testing.assert_allclose(after[:3, 3], before[:3, 3])
    editor.scene_set_transform(app, path="/group/box", rotation_wxyz=[1, 0, 0, 0])
    after = np.asarray(prim.compute_local_matrix())
    np.testing.assert_allclose(after[:3, :3], np.diag([4, 6, 8]))
    np.testing.assert_allclose(after[:3, 3], [1, 2, 3])


@pytest.mark.parametrize("kind", ["shear", "reflection", "singular"])
def test_translation_preserves_non_trs_matrix_and_partial_rotation_rejects(
    fixture, kind
):
    app, root, prim, editor = fixture
    matrix = np.eye(4, dtype=np.float32)
    if kind == "shear":
        matrix[0, 1] = 0.5
    else:
        matrix[0, 0] = -1 if kind == "reflection" else 0
    prim.set_local_matrix(matrix.T.copy())
    editor.scene_set_transform(app, path="/group/box", translation=[2, 3, 4])
    before = np.asarray(prim.compute_local_matrix()).copy()
    np.testing.assert_array_equal(before[:3, :3], matrix[:3, :3])
    for fields in ({"scale": [2, 2, 2]}, {"rotation_wxyz": [1, 0, 0, 0]}):
        with pytest.raises(AutomationError, match="transform"):
            editor.scene_set_transform(
                app, path="/group/box", translation=[9, 9, 9], **fields
            )
        np.testing.assert_array_equal(prim.compute_local_matrix(), before)
    editor.scene_set_transform(
        app, path="/group/box", rotation_wxyz=[1, 0, 0, 0], scale=[1, 1, 1]
    )
    np.testing.assert_array_equal(
        np.asarray(prim.compute_local_matrix())[:3, 3], [2, 3, 4]
    )


def test_empty_transform_edit_is_rejected(fixture):
    app, root, prim, editor = fixture
    with pytest.raises(AutomationError, match="at least one"):
        editor.scene_set_transform(app, path="/group/box")


@pytest.mark.parametrize(
    "arguments",
    [
        {"translation": [float("nan"), 0, 0]},
        {"scale": [0, 1, 1]},
        {"scale": [-1, 1, 1]},
        {"rotation_wxyz": [0, 0, 0, 0]},
        {"rotation_wxyz": [1, 0, 0]},
    ],
)
def test_invalid_transform_does_not_partially_move(fixture, arguments):
    app, root, prim, editor = fixture
    before = np.asarray(prim.compute_local_matrix()).copy()
    with pytest.raises(AutomationError):
        editor.scene_set_transform(
            app, path="/group/box", **{"translation": [1, 2, 3], **arguments}
        )
    np.testing.assert_array_equal(prim.compute_local_matrix(), before)


@pytest.mark.parametrize("owner", ["external", "articulation", "collision", "children"])
def test_reject_transform_ownership_conflicts(fixture, owner):
    app, root, prim, editor = fixture
    if owner == "external":
        prim.add_render_component().transform_source = (
            ke.render.TransformSource.EXTERNAL_BUFFER
        )
    elif owner == "articulation":
        prim.get_parent().add_articulation_component()
    elif owner == "collision":
        prim.add_collision_shape_component()
    else:
        prim.add_child("child", ke.scene.PrimType.MESH)
    with pytest.raises(AutomationError, match="Physics|leaf"):
        editor.scene_set_transform(app, path="/group/box", translation=[1, 2, 3])


def test_existing_objects_cannot_be_deleted(fixture):
    app, root, prim, editor = fixture
    with pytest.raises(AutomationError, match="automation-created"):
        editor.scene_delete(app, path="/group/box")
    assert root.get_prim_at_path("/group/box") is not None


def test_reused_path_does_not_grant_deletion_ownership(fixture):
    app, root, prim, editor = fixture
    previous = ke.scene.Prim("old", ke.scene.PrimType.MESH)
    editor._objects["/group/box"] = previous.get_transform_component()
    assert editor._objects["/group/box"].attached
    with pytest.raises(AutomationError, match="automation-created"):
        editor.scene_delete(app, path="/group/box")


def test_failed_material_replacement_preserves_color_multiplier(fixture):
    app, root, prim, editor = fixture
    prim.add_render_component()
    prim.set_display_color_alpha([0.2, 0.3, 0.4, 1])
    app.create_phong_material = lambda **kwargs: ke.material.PhongMaterial()
    app.get_scene_render_system = lambda: SimpleNamespace(
        set_material=lambda *args: False
    )
    with pytest.raises(AutomationError, match="previous material restored"):
        editor.scene_set_material(app, path="/group/box", color=[0, 1, 0])
    np.testing.assert_allclose(prim.get_display_color_alpha().x, 0.2)


@pytest.mark.parametrize(
    "path", ["/group/box", "/.Resources/x", "/bad/../x", "/missing/x", "/group/box/x"]
)
def test_invalid_creation_rejected_before_allocating_resources(fixture, path):
    app, root, prim, editor = fixture
    with pytest.raises(AutomationError):
        editor.scene_create_primitive(app, path=path)
    assert not editor._meshes and not editor._materials


@pytest.mark.parametrize("size", [(800, 200), (200, 800)])
def test_automatic_focus_fits_transformed_primitive_in_viewport(fixture, size):
    app, root, prim, editor = fixture
    mesh = ke.geometry.create_box_data(1, 1, 1)
    prim.set_mesh_data(mesh)
    editor._objects[prim.get_path()] = prim.get_transform_component()
    editor._meshes["box"] = mesh
    editor._mesh_bounds["box"] = np.array([[-0.5] * 3, [0.5] * 3])
    pose = {"position": np.array([4.0, 3.0, 4.0]), "target": np.zeros(3)}
    app.get_camera = lambda: SimpleNamespace(
        get_camera_pos=lambda: ke.Vec3(*pose["position"]),
        get_target_pos=lambda: ke.Vec3(*pose["target"]),
        get_fov=lambda: 45,
        get_near_plane=lambda: 0.1,
        get_far_plane=lambda: 1000,
    )
    app.get_width = lambda: size[0]
    app.get_height = lambda: size[1]
    app.active_scene_camera_path = lambda: None
    app.has_active_scene_camera = lambda: False
    app.set_camera_view = lambda **values: pose.update(values)
    editor.scene_set_transform(
        app,
        path=prim.get_path(),
        translation=[3, 2, 1],
        rotation_wxyz=[1, 0, 1, 0],
        scale=[2, 4, 8],
    )
    result = editor.camera_focus(app, path=prim.get_path())
    assert result["focus_mode"] == "bounds"
    np.testing.assert_allclose(result["target"], [13, 2, 1])
    position, target = pose["position"], pose["target"]
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    for vertex in mesh.vertices:
        world = np.asarray(prim.compute_world_matrix()) @ [
            vertex.x,
            vertex.y,
            vertex.z,
            1,
        ]
        offset = world[:3] - position
        depth = offset @ forward
        assert 0.1 < depth < 1000
        assert abs(offset @ up) < depth * np.tan(np.deg2rad(22.5))
        assert (
            abs(offset @ right) < depth * np.tan(np.deg2rad(22.5)) * size[0] / size[1]
        )
    # Replacing source geometry must not reuse the primitive's cached bounds.
    prim.set_mesh_data(ke.geometry.create_sphere_data(10, 8, 4))
    before = pose["position"].copy()
    with pytest.raises(AutomationError, match="Automatic focus"):
        editor.camera_focus(app, path=prim.get_path())
    np.testing.assert_array_equal(pose["position"], before)
    assert (
        editor.camera_focus(app, path=prim.get_path(), distance=5)["focus_mode"]
        == "origin"
    )

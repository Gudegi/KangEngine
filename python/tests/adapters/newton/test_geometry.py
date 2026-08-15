from pathlib import Path

import numpy as np

from kangengine.app.application import _sphere_instance_data
from kangengine.adapters.newton.conventions import (
    transform_array_to_glm_matrices,
)
from kangengine.adapters.newton.geometry import compute_vertex_normals
from kangengine.adapters.newton.buffers import (
    rgba_array,
    transform_array_to_torch_matrices,
    transform_array_to_warp_matrices,
)
from kangengine.utils.sim_buffer import to_external_transform_desc


class _CaptureViewer:
    """Create a Newton ViewerNull that records its geometry contract."""

    @staticmethod
    def create():
        import newton

        class CaptureViewer(newton.viewer.ViewerNull):
            def __init__(self):
                super().__init__()
                self.meshes = {}
                self.instances = {}

            def log_mesh(
                self,
                name,
                points,
                indices,
                normals=None,
                uvs=None,
                texture=None,
                hidden=False,
                backface_culling=True,
                **kwargs,
            ):
                del indices, normals, uvs, texture, hidden, backface_culling, kwargs
                self.meshes[str(name)] = np.asarray(points.numpy()).reshape(-1, 3)

            def log_instances(
                self,
                name,
                mesh,
                xforms,
                scales,
                colors,
                materials,
                hidden=False,
            ):
                del colors, materials
                if xforms is not None and not hidden:
                    self.instances[str(name)] = (
                        str(mesh),
                        np.asarray(xforms.numpy()).reshape(-1, 7),
                        np.asarray(scales.numpy()).reshape(-1, 3),
                    )

        return CaptureViewer()


def _find_by_mesh_token(viewer, token):
    return next(
        value
        for value in viewer.instances.values()
        if f"/{token}_" in value[0]
    )


def _compose_xyzw(parent, local):
    parent = np.asarray(parent, dtype=np.float32)
    local = np.asarray(local, dtype=np.float32)
    parent_xyz = parent[3:6]
    parent_w = parent[6]
    local_xyz = local[3:6]
    local_w = local[6]
    rotated_local_position = local[:3] + 2.0 * np.cross(
        parent_xyz,
        np.cross(parent_xyz, local[:3]) + parent_w * local[:3],
    )
    quaternion = np.concatenate(
        (
            parent_w * local_xyz
            + local_w * parent_xyz
            + np.cross(parent_xyz, local_xyz),
            [parent_w * local_w - np.dot(parent_xyz, local_xyz)],
        )
    )
    return np.concatenate((parent[:3] + rotated_local_position, quaternion))


def test_transform_array_uses_newton_xyzw_and_glm_storage():
    half_sqrt = np.sqrt(0.5)
    matrices = transform_array_to_glm_matrices(
        [[1.0, 2.0, 3.0, 0.0, 0.0, half_sqrt, half_sqrt]],
        [[2.0, 3.0, 4.0]],
    )

    expected = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(matrices[0], expected, atol=1.0e-6)


def test_numpy_transform_buffer_preserves_storage_and_version():
    matrices = np.zeros((2, 4, 4), dtype=np.float32)
    descriptor, buffer = to_external_transform_desc(matrices, version=7)

    assert buffer.data is matrices
    assert buffer.owner is matrices
    assert descriptor.count == 2
    assert descriptor.view.ptr == matrices.__array_interface__["data"][0]
    assert descriptor.view.shape == [2, 4, 4]
    assert descriptor.view.strides == [16, 4, 1]
    assert descriptor.view.version == 7


def test_compute_vertex_normals_for_xy_triangle():
    normals = compute_vertex_normals(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([0, 1, 2], dtype=np.uint32),
    )

    np.testing.assert_allclose(
        normals,
        np.array([[0, 0, 1]] * 3, dtype=np.float32),
        atol=1.0e-6,
    )


def test_empty_debug_batch_accepts_a_single_default_color():
    colors = rgba_array((0.0, 1.0, 0.0), 0, allow_cuda_readback=False)

    assert colors.shape == (0, 4)


def test_debug_colors_accept_rgb_and_rgba():
    rgb = rgba_array((0.1, 0.2, 0.3), 2, allow_cuda_readback=False)
    rgba = rgba_array((0.1, 0.2, 0.3, 0.4), 2, allow_cuda_readback=False)

    np.testing.assert_allclose(
        rgb,
        [[0.1, 0.2, 0.3, 1.0], [0.1, 0.2, 0.3, 1.0]],
    )
    np.testing.assert_allclose(
        rgba,
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
    )


def test_debug_sphere_update_preserves_colors_when_omitted():
    transforms, colors = _sphere_instance_data([[1.0, 2.0, 3.0]], 0.5, None)

    assert transforms.shape == (1, 4, 4)
    assert colors is None


def test_warp_torch_transform_conversion_matches_cpu_conversion():
    import warp as wp

    half_sqrt = np.sqrt(0.5)
    xforms = wp.array(
        [[1.0, 2.0, 3.0, 0.0, 0.0, half_sqrt, half_sqrt]],
        dtype=wp.transform,
        device="cpu",
    )
    scales = wp.array([[2.0, 3.0, 4.0]], dtype=wp.vec3, device="cpu")

    torch_matrices, stream = transform_array_to_torch_matrices(xforms, scales)
    numpy_matrices = transform_array_to_glm_matrices(xforms.numpy(), scales.numpy())

    assert stream is None
    np.testing.assert_allclose(torch_matrices.numpy(), numpy_matrices, atol=1.0e-6)

    fused_matrices, fused_stream = transform_array_to_warp_matrices(xforms, scales)
    assert fused_stream is None
    np.testing.assert_allclose(fused_matrices.numpy(), numpy_matrices, atol=1.0e-6)


def test_warp_cuda_transform_conversion_matches_cpu_conversion():
    import warp as wp

    if not wp.is_cuda_available():
        return

    half_sqrt = np.sqrt(0.5)
    transform_values = np.array(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, half_sqrt, half_sqrt],
            [-4.0, 5.0, -6.0, half_sqrt, 0.0, 0.0, half_sqrt],
        ],
        dtype=np.float32,
    )
    scale_values = np.array(
        [[2.0, 3.0, 4.0], [0.5, 1.5, 2.5]], dtype=np.float32
    )
    xforms = wp.array(transform_values, dtype=wp.transform, device="cuda:0")
    scales = wp.array(scale_values, dtype=wp.vec3, device="cuda:0")

    torch_matrices, stream = transform_array_to_torch_matrices(xforms, scales)
    fused_matrices, fused_stream = transform_array_to_warp_matrices(xforms, scales)
    numpy_matrices = transform_array_to_glm_matrices(
        transform_values, scale_values
    )

    assert torch_matrices.device.type == "cuda"
    assert stream is not None
    assert fused_matrices.device.type == "cuda"
    assert fused_stream is not None
    np.testing.assert_allclose(
        torch_matrices.cpu().numpy(), numpy_matrices, atol=1.0e-6
    )
    np.testing.assert_allclose(
        fused_matrices.cpu().numpy(), numpy_matrices, atol=1.0e-6
    )


def test_newton_primitive_mesh_bounds_are_baked_once():
    import newton

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.5)
    builder.add_shape_box(body, hx=0.5, hy=0.35, hz=0.25)
    builder.add_shape_capsule(body, radius=0.3, half_height=0.7)
    builder.add_shape_cylinder(body, radius=0.35, half_height=0.65)
    builder.add_shape_cone(body, radius=0.5, half_height=0.65)
    model = builder.finalize(device="cpu")

    viewer = _CaptureViewer.create()
    viewer.set_model(model)
    viewer.log_state(model.state())

    expected_bounds = {
        "sphere": ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
        "box": ([-0.5, -0.35, -0.25], [0.5, 0.35, 0.25]),
        "capsule": ([-0.3, -0.3, -1.0], [0.3, 0.3, 1.0]),
        "cylinder": ([-0.35, -0.35, -0.65], [0.35, 0.35, 0.65]),
        "cone": ([-0.5, -0.5, -0.65], [0.5, 0.5, 0.65]),
    }
    for token, (minimum, maximum) in expected_bounds.items():
        mesh_name, _, scales = _find_by_mesh_token(viewer, token)
        points = viewer.meshes[mesh_name]
        np.testing.assert_allclose(points.min(axis=0), minimum, atol=1.0e-5)
        np.testing.assert_allclose(points.max(axis=0), maximum, atol=1.0e-5)
        np.testing.assert_allclose(scales, 1.0, atol=1.0e-6)


def test_shape_local_transform_composes_with_body_and_static_stays_world_space():
    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    quarter_turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
    builder = newton.ModelBuilder()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(10.0, 20.0, 30.0), quarter_turn)
    )
    builder.add_shape_sphere(
        body,
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        radius=0.5,
    )
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(3.0, 4.0, 5.0), wp.quat_identity()),
        hx=1.0,
        hy=2.0,
        hz=3.0,
    )
    model = builder.finalize(device="cpu")

    viewer = _CaptureViewer.create()
    viewer.set_model(model)
    viewer.log_state(model.state())

    _, sphere_xforms, _ = _find_by_mesh_token(viewer, "sphere")
    _, box_xforms, _ = _find_by_mesh_token(viewer, "box")
    np.testing.assert_allclose(sphere_xforms[0, :3], [10.0, 21.0, 30.0])
    np.testing.assert_allclose(box_xforms[0, :3], [3.0, 4.0, 5.0])


def test_world_offsets_move_world_shapes_but_not_global_static_shapes():
    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    world_builder = newton.ModelBuilder()
    body = world_builder.add_body()
    world_builder.add_shape_sphere(body, radius=0.5)

    builder = newton.ModelBuilder()
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(2.0, 3.0, 4.0), wp.quat_identity()),
        hx=0.5,
        hy=0.5,
        hz=0.5,
    )
    model = builder.finalize(device="cpu")

    viewer = _CaptureViewer.create()
    viewer.set_model(model)
    viewer.set_world_offsets((7.0, 0.0, 0.0))
    viewer.log_state(model.state())

    _, sphere_xforms, _ = _find_by_mesh_token(viewer, "sphere")
    _, box_xforms, _ = _find_by_mesh_token(viewer, "box")
    np.testing.assert_allclose(sphere_xforms[:, :3], [[-3.5, 0.0, 0.0], [3.5, 0.0, 0.0]])
    np.testing.assert_allclose(box_xforms[0, :3], [2.0, 3.0, 4.0])


def test_mjcf_articulation_shape_transforms_match_body_local_composition():
    import newton
    import warp as wp

    newton.use_coord_layout_targets = True
    mjcf_path = (
        Path(__file__).resolve().parents[4]
        / "assets/characters/humanoid/nv_humanoid.xml"
    )
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(mjcf_path),
        floating=True,
        ignore_names=["floor", "ground"],
        xform=wp.transform((0.25, -0.5, 1.0), wp.quat_identity()),
        enable_self_collisions=False,
    )
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    viewer = _CaptureViewer.create()
    viewer.set_model(model)
    viewer.show_collision = True
    viewer.show_visual = True
    viewer.show_static = True
    viewer.show_ground = True
    viewer.log_state(state)

    body_q = np.asarray(state.body_q.numpy()).reshape(-1, 7)
    shape_body = np.asarray(model.shape_body.numpy()).reshape(-1)
    shape_local = np.asarray(model.shape_transform.numpy()).reshape(-1, 7)
    checked_shapes = 0
    expected_rendered_shapes = 0
    for batch in viewer._shape_instances.values():
        if not viewer._should_show_shape(batch.flags, batch.static):
            continue
        expected_rendered_shapes += len(batch.model_shapes)
        _, actual_xforms, actual_scales = viewer.instances[batch.name]
        expected_xforms = []
        for shape_index in batch.model_shapes:
            body_index = int(shape_body[shape_index])
            expected_xforms.append(
                shape_local[shape_index]
                if body_index < 0
                else _compose_xyzw(body_q[body_index], shape_local[shape_index])
            )
        expected_matrices = transform_array_to_glm_matrices(
            expected_xforms, actual_scales
        )
        actual_matrices = transform_array_to_glm_matrices(
            actual_xforms, actual_scales
        )
        np.testing.assert_allclose(actual_matrices, expected_matrices, atol=2.0e-5)
        checked_shapes += len(expected_xforms)

    assert checked_shapes == expected_rendered_shapes
    assert checked_shapes > 0

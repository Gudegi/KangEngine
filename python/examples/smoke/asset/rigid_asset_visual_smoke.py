"""Render shared rigid assets, independent resets, and textured dynamic props."""

import math
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import kangengine as ke
from kangengine.app.application import RenderablePrimView
from kangengine.asset import load_obj_rigid, rigid_from_terrain_mesh
from kangengine.sim import GridCloner, KangSimWorld
from kangengine.sim.rigid_query import RigidQueryMixin
from kangengine.terrain import pyramid_stairs_mesh
from kangengine.visual.sim import SimWorldVisualizer
from kangengine.visual.sim import world_visualizer


@contextmanager
def _direct_cuda_sync():
    with (
        patch.object(
            RigidQueryMixin,
            "get_root_pose",
            side_effect=AssertionError("CUDA visuals must consume native rigid state"),
        ),
        patch.object(
            RenderablePrimView,
            "set_transform_buffer",
            side_effect=AssertionError("CUDA visuals must write mapped GL buffers"),
        ),
    ):
        yield


@contextmanager
def _mapped_transform_readback():
    captured = []
    launch = world_visualizer._ke.rigid_state_to_mapped_mat4_cuda

    def capture(state, rows, local_transforms, mapped):
        launch(state, rows, local_transforms, mapped)

        # Inspect the actual mapped VBO before unmapping invalidates its pointer.
        for output in mapped:
            borrowed = ke.physics.GpuArrayView()
            borrowed.memory_type = ke.physics.SimMemoryType.CUDA_DEVICE
            for name in (
                "ptr",
                "dtype",
                "device_id",
                "shape",
                "strides",
                "stream_handle",
            ):
                setattr(borrowed, name, getattr(output, name))
            stream = torch.cuda.ExternalStream(
                output.stream_handle, device=output.device_id
            )
            with torch.cuda.stream(stream):
                captured.append(borrowed.torch().clone().cpu().numpy())

    with patch.object(
        world_visualizer._ke, "rigid_state_to_mapped_mat4_cuda", new=capture
    ):
        yield captured


def _standalone_visual():
    app: ke.App = ke.App()
    app.initialize(
        width=320, height=240, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True
    )
    try:
        half = math.sqrt(0.5)
        description = ke.asset.RigidDesc(
            visual_geoms=[
                ke.asset.VisualGeomDesc(
                    mesh_data=ke.geometry.create_box_data(1.0, 1.0, 1.0),
                    position=(1.0, 0.0, 0.5),
                    quat_xyzw=(0.0, half, 0.0, half),
                    scale=(0.5, 2.0, 3.0),
                )
            ]
        )
        parent = app.scene.define_prim("/parent", ke.scene.PrimType.XFORM)
        parent.set_local_translation(ke.Vec3(8.0, 5.0, 2.0))
        parent.set_local_quat_wxyz(ke.Quat.from_xyzw((0.0, 0.0, half, half)))
        parent.set_local_scale(ke.Vec3(2.0, 3.0, 4.0))
        app.set_camera_view(position=(6.0, -10.0, 5.0), target=(0.0, 0.0, 1.0))
        app.draw_frame()
        background = np.asarray(app.read_rgb_pixels(True)).copy()

        # Source geometry can create independent visuals without a physics world.
        first = ke.visual.RigidVisual(app, "/parent/first", description)
        second = ke.visual.RigidVisual(app, "/second", description)
        first.apply_pose(position=(3.0, -2.0, 1.0), quat_xyzw=(0.0, 0.0, half, half))
        second.apply_pose(position=(-2.0, 0.0, 1.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0))
        first_prim, second_prim = first.body_prims[0], second.body_prims[0]
        assert (
            first_prim.get_render_component().transform_source
            == ke.render.TransformSource.SCENE_GRAPH
        )
        first_resource = first_prim.get_mesh_component().resource_handle
        assert first_resource != ke.scene.InvalidResourceHandle
        assert first_resource == second_prim.get_mesh_component().resource_handle

        # A transformed parent must not reinterpret the supplied world pose as local.
        expected = np.array(
            ((0, -2, 0, 3), (0, 0, 3, -1), (-0.5, 0, 0, 1.5), (0, 0, 0, 1))
        )
        np.testing.assert_allclose(
            first_prim.compute_world_matrix(), expected, atol=1e-5, rtol=0
        )
        second_before = np.asarray(second_prim.compute_world_matrix()).copy()
        np.testing.assert_allclose(second_before[:3, 3], (-1, 0, 1.5), atol=1e-6)
        first.apply_pose(position=(4.0, -2.0, 1.0), quat_xyzw=(0.0, 0.0, half, half))
        expected[0, 3] += 1
        np.testing.assert_allclose(
            first_prim.compute_world_matrix(), expected, atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(second_prim.compute_world_matrix(), second_before)
        app.draw_frame()
        rendered = np.asarray(app.read_rgb_pixels(True))
        assert np.count_nonzero(np.any(rendered != background, axis=-1)) > 100

        first.remove()
        assert app.scene.get_prim_at_path("/parent/first") is None
        assert app.scene.get_prim_at_path("/parent") is not None
        assert second_prim.get_render_component().attached

        # Old instance cleanup must not remove a replacement at the same path.
        replacement = ke.visual.RigidVisual(app, "/parent/first", description)
        first.remove()
        assert app.scene.get_prim_at_path("/parent/first/part_0") is not None
        replacement.apply_pose(position=(0.0, 0.0, 1.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0))
        try:
            first.apply_pose(position=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0))
        except RuntimeError as error:
            assert "removed" in str(error)
        else:
            raise AssertionError("removed visuals must reject pose updates")
        replacement.remove()

        second.remove()
        assert app.scene.get_prim_at_path("/second") is None
        assert app.get_scene_render_system().registration_count == 0
        print("PASS: standalone rigid visual world pose, shared geometry, and removal")
    finally:
        app.close()


def _shared_geometry(device):
    app: ke.App = ke.App()
    app.initialize(
        width=320, height=240, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True
    )
    world: KangSimWorld = KangSimWorld(
        num_envs=2, sim_device=device, state_device=device
    )
    cloner: GridCloner = GridCloner(world, spacing=10.0)
    mesh = pyramid_stairs_mesh()
    assets = [rigid_from_terrain_mesh(mesh) for _ in range(2)]
    assets[1].visual_geoms = [
        *assets[1].visual_geoms,
        ke.asset.VisualGeomDesc(
            mesh_data=assets[0].visual_geoms[0].mesh_data,
            position=(0.0, 0.0, 0.5),
            scale=(0.25, 0.25, 0.25),
        ),
    ]
    batch = cloner.add_rigid(assets, obj_id=0, kinematic=True)
    if device == "cuda":
        world.init_gpu_system()

    viewer = SimWorldVisualizer(app, world)
    copy_viewer = SimWorldVisualizer(app, world)
    try:
        app.set_camera_view(position=(-12.0, -16.0, 13.0), target=(0.0, 0.0, 0.5))
        app.draw_frame()
        background = np.asarray(app.read_rgb_pixels(True)).copy()

        visual = viewer.add(batch, path="/rigids")
        assert len(visual.prims) == 3
        assert isinstance(
            visual._backend,
            world_visualizer.RigidGPUExternalBackend
            if device == "cuda"
            else world_visualizer.RigidCPUExternalBackend,
        )
        prim = app.scene.get_prim_at_path("/rigids/asset_0/part_0")
        assert (
            prim.get_mesh_component().resource_handle != ke.scene.InvalidResourceHandle
        )
        assert (
            prim.get_render_component().transform_source
            == ke.render.TransformSource.EXTERNAL_BUFFER
        )

        # Shared geometry must not merge independently updated transform buffers.
        copy_viewer.add(batch, path="/rigid_copy")
        render_system = app.get_scene_render_system()
        for path in (
            "/rigids/asset_1/part_0",
            "/rigid_copy/asset_0/part_0",
        ):
            other = app.scene.get_prim_at_path(path)
            assert (
                other.get_mesh_component().resource_handle
                == prim.get_mesh_component().resource_handle
            )
            assert not render_system.shares_batch(
                prim.get_render_component(), other.get_render_component()
            )
        picked = visual.pick_body(
            SimpleNamespace(hit=True, handle=visual.body_handles[1], instance_index=0)
        )
        assert picked.env_id == 1 and picked.body_id == 0

        app.draw_frame()
        before = np.asarray(app.read_rgb_pixels(True)).copy()
        foreground = np.any(before != background, axis=-1)
        assert np.count_nonzero(foreground) > 100
        # Missing UVs must not paint the whole procedural surface as black grid lines.
        assert np.median(before[foreground].mean(axis=-1)) > 40
        first_pose = batch.get_root_pose()[0][0].clone()

        # One actor moves and turns; its render instance follows the applied pose.
        position = cloner.env_origins[1:2].to(device) + torch.tensor(
            [[0.0, 0.0, 1.0]], device=device
        )
        rotation = torch.tensor(
            [[0.0, 0.0, math.sin(0.3), math.cos(0.3)]], device=device
        )
        batch.set_root_state([1], position, rotation, immediate=device == "cpu")
        if device == "cuda":
            world.step(substeps=0, refresh=False, apply_commands=False)
        viewer.sync()
        copy_viewer.sync()
        app.draw_frame()
        after = np.asarray(app.read_rgb_pixels(True))
        assert np.count_nonzero(np.any(before != after, axis=-1)) > 100
        torch.testing.assert_close(batch.get_root_pose()[0][0], first_pose)

        visual.release()
        assert app.scene.get_prim_at_path("/rigids") is None
        viewer.add(batch, path="/rigids")
        app.draw_frame()
        print(
            f"PASS ({device}): independent asset pose, shared resources, and reattachment"
        )
    finally:
        copy_viewer.release()
        viewer.release()
        world.release()
        app.close()


def _box_collision():
    shape = ke.asset.CollisionGeomDesc(
        type=ke.asset.CollisionGeomDescType.BOX,
        size=np.array([0.8, 0.3, 0.5], dtype=np.float32),
        position=np.zeros(3, dtype=np.float32),
        quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    return [shape]


def _textured_obj(directory):
    # Two material subsets share one actor but differ from its box collider.
    texture = np.full((8, 8, 3), [255, 15, 15], dtype=np.uint8)
    texture[::2, ::2] = [255, 180, 180]
    Image.fromarray(texture).save(directory / "paint.png")
    (directory / "prop.mtl").write_text(
        "newmtl Paint\nKd 1 1 1\nmap_Kd paint.png\n"
        "newmtl Green\nKd 0.02 1 0.02\nd 0.6\n",
        encoding="utf-8",
    )
    obj_path = directory / "prop.obj"
    obj_path.write_text(
        "mtllib prop.mtl\n"
        "v -1.2 -0.5 -0.5\nv 0 -0.5 -0.5\nv 0 -0.5 0.5\nv -1.2 -0.5 0.5\n"
        "v 0 -0.5 -0.5\nv 1.2 -0.5 -0.5\nv 1.2 -0.5 0.5\nv 0 -0.5 0.5\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 -1 0\n"
        "usemtl Paint\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"
        "usemtl Green\nf 5/1/1 6/2/1 7/3/1\nf 5/1/1 7/3/1 8/4/1\n",
        encoding="utf-8",
    )
    return obj_path


def _textured_dynamic(device):
    with tempfile.TemporaryDirectory() as directory:
        asset = load_obj_rigid(
            _textured_obj(Path(directory)), collision_geoms=_box_collision()
        )
        app: ke.App = ke.App()
        app.initialize(
            width=320, height=240, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True
        )
        world = KangSimWorld(
            num_envs=1,
            sim_device=device,
            state_device=device,
            sim_dt=1.0 / 120.0,
            add_ground=True,
        )
        batch = GridCloner(world).add_rigid(
            asset, initial_root_pos=(0.0, 0.0, 2.0), density=600.0
        )
        if device == "cuda":
            world.init_gpu_system()

        viewer = SimWorldVisualizer(app, world)
        try:
            app.set_camera_view(position=(0.0, -6.0, 3.0), target=(0.0, 0.0, 1.0))
            app.set_light_direction(ke.Vec3(0.0, -1.0, 0.0))
            visual = viewer.add(batch, path="/prop")
            assert isinstance(
                visual._backend,
                world_visualizer.RigidGPUExternalBackend
                if device == "cuda"
                else world_visualizer.RigidCPUExternalBackend,
            )
            assert len(visual.prims) == 2
            assert (
                visual.prims[1].get_render_component().alpha_mode
                == ke.render.AlphaMode.BLEND
            )
            np.testing.assert_allclose(
                visual._backend._groups[0][1].base_colors[:, 3], [1.0, 0.6]
            )
            for prim in visual.prims:
                assert (
                    prim.get_mesh_component().resource_handle
                    != ke.scene.InvalidResourceHandle
                )

            app.draw_frame()
            before = np.asarray(app.read_rgb_pixels(True)).copy()
            red = before[..., 0].astype(float)
            green = before[..., 1].astype(float)
            assert np.count_nonzero((red > 1.5 * green) & (red > 40)) > 100
            green_pixels = np.count_nonzero((green > 1.5 * red) & (green > 40))
            assert green_pixels > 100

            # MTL opacity combines with instance tint in the external color buffer.
            visual.set_color((1.0, 1.0, 1.0, 0.25))
            app.draw_frame()
            faded = np.asarray(app.read_rgb_pixels(True)).astype(float)
            faded_green = (faded[..., 1] > 1.5 * faded[..., 0]) & (faded[..., 1] > 40)
            assert np.count_nonzero(faded_green) < green_pixels
            visual.set_color((1.0, 1.0, 1.0, 1.0))
            initial_position = batch.get_root_pose()[0].clone()

            # Dynamics changes the root pose; both visual material parts follow it.
            world.step(substeps=90, refresh=True)
            with _direct_cuda_sync() if device == "cuda" else nullcontext():
                viewer.sync()
            app.draw_frame()
            after = np.asarray(app.read_rgb_pixels(True))
            assert batch.get_root_pose()[0][0, 2] < initial_position[0, 2] - 0.5
            assert np.count_nonzero(np.any(before != after, axis=-1)) > 100

            if device == "cpu":
                inspection = viewer.add_scene_graph(
                    batch, env_id=0, path="/inspect", color=(1.0, 1.0, 1.0, 0.5)
                )
                assert math.isclose(
                    inspection.body_prims[1].get_display_color_alpha().w,
                    0.3,
                    abs_tol=1e-6,
                )
                inspection.set_color((1.0, 1.0, 1.0, 0.25))
                assert math.isclose(
                    inspection.body_prims[1].get_display_color_alpha().w,
                    0.15,
                    abs_tol=1e-6,
                )
                viewer.sync()
                root = batch.get_root_pose()[0][0]
                for prim in inspection.body_prims:
                    position = prim.get_world_translation()
                    np.testing.assert_allclose(
                        [position.x, position.y, position.z], root.numpy(), atol=1e-6
                    )
            print(
                f"PASS ({device}): textured multipart dynamic rigid with separate collider"
            )
        finally:
            viewer.release()
            world.release()
            app.close()


def _compound_pose(device):
    app: ke.App = ke.App()
    app.initialize(
        width=320, height=240, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True
    )
    world = KangSimWorld(num_envs=1, sim_device=device, state_device=device)
    half = math.sqrt(0.5)
    local_poses = (
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.5), (0.0, half, 0.0, half)),
    )
    shapes = [
        ke.asset.CollisionGeomDesc(
            type=ke.asset.CollisionGeomDescType.BOX,
            size=np.array([0.5, 0.25, 0.2], dtype=np.float32),
            position=np.array(position, dtype=np.float32),
            quat_xyzw=rotation,
        )
        for position, rotation in local_poses
    ]
    mesh = ke.geometry.create_box_data(1.0, 1.0, 1.0)
    parts = [
        ke.asset.VisualGeomDesc(
            mesh_data=mesh,
            position=position,
            quat_xyzw=rotation,
            scale=scale,
        )
        for (position, rotation), scale in zip(
            local_poses, ((2.0, 0.5, 1.5), (0.5, 2.0, 3.0))
        )
    ]
    descriptor = ke.asset.RigidDesc(
        name="compound", visual_geoms=parts, collision_geoms=shapes
    )
    batch = GridCloner(world).add_rigid(
        descriptor,
        initial_root_pos=(3.0, -2.0, 1.0),
        initial_root_quat_xyzw=(0.0, 0.0, half, half),
        kinematic=True,
    )
    fixed = GridCloner(world).add_rigid(
        ke.asset.RigidDesc(collision_geoms=_box_collision()),
        obj_id=1,
        initial_root_pos=(0.0, 0.0, -2.0),
        static=True,
    )
    if device == "cuda":
        world.init_gpu_system()

    viewer = SimWorldVisualizer(app, world)
    try:
        app.set_camera_view(position=(0.0, -8.0, 5.0), target=(3.0, -2.0, 1.0))
        stream = torch.cuda.Stream() if device == "cuda" else None
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            with _direct_cuda_sync() if device == "cuda" else nullcontext():
                visual = viewer.add(batch, path="/compound")
        if device == "cuda":
            assert isinstance(visual._backend, world_visualizer.RigidGPUExternalBackend)
            # Row readiness and output ownership must survive a stream change.
            next_stream = torch.cuda.Stream()
            with (
                torch.cuda.stream(next_stream),
                _direct_cuda_sync(),
                _mapped_transform_readback() as transforms,
            ):
                visual.sync()
            torch.cuda.current_stream().wait_stream(next_stream)
        else:
            assert isinstance(visual._backend, world_visualizer.RigidCPUExternalBackend)
            native_batch = visual._backend._batches[0][2]
            transforms = [
                np.asarray(native_batch.transforms(part_id)).reshape(-1, 4, 4)
                for part_id in range(2)
            ]
        assert len(visual.prims) == 2

        # Column-major matrices include actor rotation, local pose, and local scale.
        expected = (
            ((0, 2, 0, 0), (-0.5, 0, 0, 0), (0, 0, 1.5, 0), (3, -3, 1, 1)),
            ((0, 0, -0.5, 0), (-2, 0, 0, 0), (0, 3, 0, 0), (3, -1, 1.5, 1)),
        )
        assert len(transforms) == len(expected)
        for actual, matrix in zip(transforms, expected):
            np.testing.assert_allclose(actual[0], matrix, atol=1e-6, rtol=0)

        # Static actors have no GPU rigid-state row, even in a CUDA world.
        fixed_visual = viewer.add(fixed, path="/fixed")
        assert isinstance(
            fixed_visual._backend, world_visualizer.RigidCPUExternalBackend
        )
        native_batch = fixed_visual._backend._batches[0][2]
        np.testing.assert_allclose(native_batch.transforms(0)[0][12:15], (0, 0, -2))
        app.draw_frame()
        print(f"PASS ({device}): compound local pose/scale and static backend routing")
    finally:
        viewer.release()
        world.release()
        app.close()


def _textured_articulation(device):
    with tempfile.TemporaryDirectory() as directory:
        source = load_obj_rigid(_textured_obj(Path(directory)))
        tree = ke.animation.SkeletonTree(
            ["base", "link"],
            [-1, 0],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]),
            np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        )
        joint = ke.asset.JointDesc(
            "hinge",
            type=ke.asset.JointDescType.REVOLUTE,
            axis=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            joint_offset=np.zeros(3, dtype=np.float32),
            lo_limit=-1.0,
            hi_limit=1.0,
        )
        parts = [
            ke.asset.VisualGeomDesc(
                mesh_data=part.mesh_data,
                material=part.material,
                body_index=body_id,
                body_name=name,
                rgba=part.rgba,
            )
            for body_id, name in enumerate(("base", "link"))
            for part in source.visual_geoms
        ]
        descriptor = ke.asset.ArticulationDesc(
            tree,
            joints={1: [joint]},
            collision_geoms={0: _box_collision(), 1: _box_collision()},
            inertials={},
            visual_geoms=parts,
            asset_dir=source.asset_dir,
        )
        app: ke.App = ke.App()
        app.initialize(
            width=320, height=240, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True
        )
        world = KangSimWorld(num_envs=1, sim_device=device, state_device=device)
        batch = GridCloner(world).add_articulation(
            descriptor, initial_root_pos=(0.0, 0.0, 1.0)
        )
        if device == "cuda":
            world.init_gpu_system()

        viewer = SimWorldVisualizer(app, world)
        try:
            app.set_camera_view(position=(0.0, -7.0, 3.5), target=(0.0, 0.0, 1.5))
            app.set_light_direction(ke.Vec3(0.0, -1.0, 0.0))
            visual = viewer.add(batch, path="/articulation")
            assert len(visual.body_handles) == 4
            assert [
                visual.body_id_from_render_handle(handle)
                for handle in visual.body_handles
            ] == [0, 0, 1, 1]
            app.draw_frame()
            before = np.asarray(app.read_rgb_pixels(True)).copy()
            red = before[..., 0].astype(float)
            green = before[..., 1].astype(float)
            assert np.count_nonzero((red > 1.5 * green) & (red > 40)) > 100
            assert np.count_nonzero((green > 1.5 * red) & (green > 40)) > 100

            batch.set_root_state(
                [0],
                torch.tensor([[0.5, 0.0, 1.0]], device=device),
                torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device),
                immediate=device == "cpu",
            )
            if device == "cuda":
                world.step(substeps=0, refresh=False, apply_commands=False)
            viewer.sync()
            app.draw_frame()
            after = np.asarray(app.read_rgb_pixels(True))
            assert np.count_nonzero(np.any(before != after, axis=-1)) > 100
            print(
                f"PASS ({device}): textured articulation visual source from descriptor"
            )
        finally:
            viewer.release()
            world.release()
            app.close()


def _shared_pbr_material(device):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pixels = {
            "base_color_texture": (240, 200, 160),
            "normal_texture": (128, 128, 255),
            "metallic_roughness_texture": (255, 140, 70),
            "metallic_texture": (90, 90, 90),
            "roughness_texture": (160, 160, 160),
            "ao_texture": (220, 220, 220),
            "orm_texture": (240, 180, 70),
            "emissive_texture": (10, 20, 30),
        }
        for slot, color in pixels.items():
            Image.new("RGB", (4, 4), color).save(root / f"{slot}.png")

        # Material factors and paths are authored before creating an OpenGL app.
        source = ke.material.PBRMaterial()
        source.base_color = ke.Vec4(0.8, 0.3, 0.15, 0.8)
        source.metallic = 0.23
        source.roughness = 0.41
        source.emissive_color = ke.Vec3(0.01, 0.02, 0.03)
        source.emissive_strength = 2.5
        source.texture_paths = {slot: f"{slot}.png" for slot in pixels}
        part = ke.asset.VisualGeomDesc(
            mesh_data=ke.geometry.create_box_data(1.0, 1.0, 1.0),
            material=source,
            rgba=(1.0, 1.0, 1.0, 0.5),
        )
        rigid_desc = ke.asset.RigidDesc(
            visual_geoms=[part], collision_geoms=_box_collision(), asset_dir=directory
        )
        tree = ke.animation.SkeletonTree(
            ["body"], [-1], np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]])
        )
        articulated_desc = ke.asset.ArticulationDesc(
            tree,
            joints={},
            collision_geoms={0: _box_collision()},
            inertials={},
            visual_geoms=[part],
            asset_dir=directory,
        )
        assert all(getattr(source, slot) is None for slot in pixels)

        app: ke.App = ke.App()
        app.initialize(
            width=320, height=240, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True
        )
        world = KangSimWorld(num_envs=1, sim_device=device, state_device=device)
        cloner = GridCloner(world)
        rigid = cloner.add_rigid(
            rigid_desc, obj_id=0, kinematic=True, initial_root_pos=(-1.0, 0.0, 1.0)
        )
        articulated = cloner.add_articulation(
            articulated_desc, obj_id=1, initial_root_pos=(1.0, 0.0, 1.0)
        )
        if device == "cuda":
            world.init_gpu_system()

        viewer = SimWorldVisualizer(app, world)
        try:
            app.set_camera_view(position=(0.0, -6.0, 3.0), target=(0.0, 0.0, 1.0))
            app.set_light_direction(ke.Vec3(0.0, -1.0, 0.3))
            tint = (1.0, 1.0, 1.0, 0.5)
            rigid_visual = viewer.add(rigid, path="/pbr_rigid", color=tint)
            articulated_visual = viewer.add(articulated, path="/pbr_artic", color=tint)
            render_prims = (
                rigid_visual.prims[0],
                app.scene.get_prim_at_path("/pbr_artic/body/visual_0"),
            )
            materials = [prim.get_material() for prim in render_prims]
            assert materials[0] is not materials[1]
            for prim, runtime in zip(render_prims, materials):
                assert isinstance(runtime, ke.material.NativePBRMaterial)
                assert runtime is not source.native
                np.testing.assert_allclose(runtime.base_color, source.base_color)
                np.testing.assert_allclose(
                    runtime.emissive_color, source.emissive_color
                )
                for factor in ("metallic", "roughness", "emissive_strength"):
                    assert math.isclose(
                        getattr(runtime, factor), getattr(source, factor)
                    )
                assert (
                    prim.get_render_component().alpha_mode == ke.render.AlphaMode.BLEND
                )
                for slot in pixels:
                    assert getattr(runtime, slot) is not None

            for slot in pixels:
                assert getattr(source, slot) is None
                assert getattr(materials[0], slot) is getattr(materials[1], slot)
            # PBR alpha stays in the material; the instance carries only the part tint.
            assert math.isclose(
                rigid_visual._backend._groups[0][1].base_colors[0, 3], 0.5
            )
            assert math.isclose(articulated_visual._backend._base_colors[0, 3], 0.5)
            app.draw_frame()
            before = np.asarray(app.read_rgb_pixels(True)).copy()
            for visual in (rigid_visual, articulated_visual):
                visual.set_color((1.0, 1.0, 1.0, 1.0))
            app.draw_frame()
            after = np.asarray(app.read_rgb_pixels(True))
            assert np.count_nonzero(np.any(before != after, axis=-1)) > 100

            if device == "cpu":
                inspections = (
                    viewer.add_scene_graph(
                        rigid, env_id=0, path="/inspect_rigid", color=tint
                    ),
                    viewer.add_scene_graph(
                        articulated, env_id=0, path="/inspect_artic", color=tint
                    ),
                )
                for inspection in inspections:
                    prim = inspection.prims[0]
                    assert math.isclose(prim.get_display_color_alpha().w, 0.25)
                    assert math.isclose(
                        prim.get_material().base_color.w, 0.8, abs_tol=1e-6
                    )
                    inspection.set_color((1.0, 1.0, 1.0, 0.25))
                    assert math.isclose(prim.get_display_color_alpha().w, 0.125)
            print(f"PASS ({device}): shared PBR factors, deferred maps, and alpha tint")
        finally:
            viewer.release()
            world.release()
            app.close()


def main(device="cpu"):
    _shared_geometry(device)
    _textured_dynamic(device)
    _compound_pose(device)
    _textured_articulation(device)
    _shared_pbr_material(device)


if __name__ == "__main__":
    _standalone_visual()
    main()
    if torch.cuda.is_available():
        main("cuda")

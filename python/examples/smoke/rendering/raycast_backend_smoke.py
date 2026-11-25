"""Check world-space mesh hits and CPU instance ownership on both backends."""

import argparse
from unittest.mock import patch

import numpy as np

import kangengine as ke

from skinning_backend_smoke import cube_asset


def check_queries(app):
    mesh = ke.geometry.create_cube_data(1)
    material = app.create_pbr_material()
    near = app.scene.add_mesh("/near", mesh, material)
    far = app.scene.add_mesh("/far", mesh, material)
    near.set_local_translation(ke.Vec3(0, 0, 2))
    app.draw_frame()

    def cast(x=0, y=0):
        return app.raycast(origin=ke.Vec3(x, y, 10), direction=ke.Vec3(0, 0, -2))

    def expect(path, distance, x=0, y=0):
        hit = cast(x, y)
        assert hit.hit and hit.prim.get_path() == path, (path, hit.hit)
        assert abs(hit.distance - distance) < 1e-4, hit.distance
        np.testing.assert_allclose([hit.position.x, hit.position.y, hit.position.z],
                                   [x, y, 10 - distance], atol=1e-4)
        return hit

    expect("/near", 7.5)
    # Selection changes apply immediately, without another rendered frame.
    near.prim.set_pickable(False)
    expect("/far", 9.5)
    near.prim.set_pickable(True)
    near.set_visible(False)
    expect("/far", 9.5)
    near.set_visible(True)
    near.prim.set_active(False)
    expect("/far", 9.5)
    near.prim.set_active(True)
    expect("/near", 7.5)
    near.remove()
    expect("/far", 9.5)
    app.draw_frame()
    app.draw_frame()
    expect("/far", 9.5)

    # Off-camera, mirrored and nonuniform transforms still report world distance.
    far.set_local_translation(ke.Vec3(20, 0, 1))
    far.prim.set_local_scale(ke.Vec3(-2, 3, 4))
    app.draw_frame()
    expect("/far", 7, x=20)
    far.prim.set_local_scale(ke.Vec3(0, 1, 1))
    app.draw_frame()
    assert not cast(20).hit
    far.prim.set_local_scale(ke.Vec3(1, 1, 1))
    far.set_local_translation(ke.Vec3(0, 0, 0))

    replacement = ke.geometry.create_cube_data(2)
    resource = app.resources.register_mesh("Replacement", replacement)
    far.prim.set_mesh_data(replacement)
    far.prim.get_mesh_component().resource_handle = resource
    app.draw_frame()
    expect("/far", 9)
    far.update_geometry(np.asarray(replacement.vertices, dtype=np.float32) * 0.8)
    app.draw_frame()
    assert not cast().hit, "Deformed meshes must not return bind-pose hits"
    far.remove()

    batch = app.scene.add_mesh(
        "/batch", mesh, material,
        transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
    )
    system = app.get_scene_render_system()
    matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    matrices[:, 3, 0] = [-2, 2]
    matrices[1, 3, 2] = 3
    system.update_instances(batch.component, matrices)
    app.draw_frame()
    first, second = cast(-2), cast(2)
    assert first.hit and second.hit and first.handle == second.handle
    assert first.prim is None and second.prim is None
    assert first.instance_index == 0 and second.instance_index == 1
    assert abs(first.distance - 9.5) < 1e-4
    assert abs(second.distance - 6.5) < 1e-4
    assert second.transform_source == ke.render.TransformSource.EXTERNAL_BUFFER
    batch.prim.set_pickable(False)
    assert not cast(2).hit
    batch.prim.set_pickable(True)
    matrices[1, 3, 2] = 5
    system.update_instances(batch.component, matrices)
    assert abs(cast(2).distance - 4.5) < 1e-4

    # Versioned CPU buffers use the last synchronized transform snapshot.
    source = ke.physics.GpuArrayView()
    source.ptr = matrices.ctypes.data
    source.memory_type = ke.physics.SimMemoryType.CPU_HOST
    source.dtype = ke.physics.SimDType.FLOAT32
    source.shape = list(matrices.shape)
    source.version = 1
    source.set_owner(matrices)
    descriptor = ke.render.ExternalBufferDesc(
        view=source, sync_policy=ke.render.ExternalSyncPolicy.VERSIONED,
    )
    system.set_external_buffer(batch.component, descriptor)
    app.draw_frame()
    matrices[1, 3, 2] = 6
    assert abs(cast(2).distance - 4.5) < 1e-4
    source.version += 1
    descriptor.view = source
    system.set_external_buffer(batch.component, descriptor)
    app.draw_frame()
    assert abs(cast(2).distance - 3.5) < 1e-4
    batch.remove()
    assert not cast(2).hit

    surface = ke.visual.SkinnedSurface.create(app, "/skin", cube_asset(), material=material)
    app.draw_frame()
    assert not cast().hit, "Skinned meshes must not return bind-pose hits"
    surface.remove()

    for direction in (ke.Vec3(0, 0, 0), ke.Vec3(float("nan"), 0, -1)):
        try:
            app.raycast(origin=ke.Vec3(0, 0, 0), direction=direction)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid ray was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = ke.App()
    try:
        app.initialize(width=128, height=96, hide_ui=True, headless=True,
                       renderer=args.renderer)
        app.set_vsync(False)
        check_queries(app)
        if args.renderer == "ovrtx":
            from kangengine.render._ovrtx_scene import OvrtxAppAdapter

            # A world query must not render or download an image.
            mesh = ke.geometry.create_cube_data(1)
            app.scene.add_mesh("/query_only", mesh, app.create_pbr_material())
            app.draw_frame()
            with patch.object(OvrtxAppAdapter, "render", side_effect=AssertionError):
                assert app.raycast(ke.Vec3(0, 0, 10), ke.Vec3(0, 0, -1)).hit
        print(f"PASS: {args.renderer} world rays, identities and CPU instances")
    finally:
        app.close()


if __name__ == "__main__":
    main()

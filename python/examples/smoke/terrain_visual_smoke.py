"""Verify terrain source-resource registration and repeated viewer attachment."""

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(
        width=64, height=64, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True
    )
    materials = app.create_standard_materials()
    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.z_up())
    asset = ke.terrain.pyramid_stairs_mesh()
    instance = asset.create(world)
    for _ in range(2):
        instance.add_visual(app.scene, "/terrain", material=materials.ground)
        for _, path in instance._prims:
            prim = app.scene.get_prim_at_path(path)
            component = prim.get_mesh_component()
            assert component.resource_handle != ke.scene.InvalidResourceHandle
            assert component.mesh_data is not None
            assert prim.get_render_component() is not None
        app.render_frame_once()
        instance.remove_visual()
        assert app.scene.get_prim_at_path("/terrain/chunk_0") is None
    instance.remove()
    assert world.num_ground_actors() == 0
    print("PASS: terrain visual resource registration and removal")


if __name__ == "__main__":
    main()

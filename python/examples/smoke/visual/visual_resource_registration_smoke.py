"""Check shared mesh identities through Native and scene-data rendering."""

from pathlib import Path

import numpy as np

import kangengine as ke


class RecordingRenderer:
    frame = None

    def render(self, frame):
        self.frame = frame
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self):
        pass


def check(native):
    app = ke.App()
    recording = RecordingRenderer()
    try:
        if native:
            app.initialize(width=32, height=32, headless=True, hide_ui=True)
        else:
            app._initialize_scene_renderer(
                32, 32, ke.UpAxis.Z, ke.scene.BackendType.NATIVE,
                lambda width, height: recording,
            )
        material = app.create_pbr_material()
        mesh = ke.geometry.create_cube_data(0.2)
        prim = app.scene.define_prim("/direct", ke.scene.PrimType.MESH)
        prim.set_mesh_data(mesh)
        direct = app.scene.add_renderable(prim, material)
        handle = prim.get_mesh_component().resource_handle
        assert handle != ke.scene.InvalidResourceHandle
        assert app.resources.mesh(handle) is mesh
        reused = app.scene.add_mesh("/reused", mesh, material)
        assert reused.prim.get_mesh_component().resource_handle == handle

        mjcf_path = Path(ke.__file__).parent / "assets/characters/kw/kw5.xml"
        surface = ke.visual.ArticulatedSurface.create_from_mjcf(
            app, "/robot", mjcf_path, material=material,
        )
        before_clone = len(app.resources)
        clone = surface.create_instance("/clone")
        assert len(app.resources) == before_clone
        assert len(surface.views) > 0
        handles = []
        for original, copied in zip(surface.views, clone.views, strict=True):
            source = original.prim.resolve_mesh_data()
            resource = original.prim.get_mesh_component().resource_handle
            assert resource != ke.scene.InvalidResourceHandle
            assert copied.prim.get_mesh_component().resource_handle == resource
            assert copied.prim.resolve_mesh_data() is source
            assert app.resources.mesh(resource) is source
            handles.append(resource)
        app.draw_frame()
        if not native:
            objects = recording.frame["objects"]
            assert len(objects) == 2 + 2 * len(surface.views)
            for item in objects:
                assert app.resources.mesh(item["mesh_id"]) is item["mesh"]

        surface.remove()
        app.draw_frame()
        for view, resource in zip(clone.views, handles, strict=True):
            assert app.resources.mesh(resource) is view.prim.resolve_mesh_data()
        clone.remove()
        direct.remove()
        reused.remove()
        app.draw_frame()
    finally:
        app.close()


if __name__ == "__main__":
    check(False)
    check(True)
    print("PASS: direct and articulated visuals share registered mesh resources")

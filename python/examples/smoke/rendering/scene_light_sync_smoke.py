"""Check all scene light types outside /lights through either App backend."""

import argparse

import numpy as np
import kangengine as ke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = ke.App()
    try:
        app.initialize(width=160, height=160, hide_ui=True, headless=True,
                       up_axis=ke.UpAxis.Z, renderer=args.renderer)
        app.set_frame_capture_active(True)
        app.scene.get_prim_at_path("/lights/default_directional").set_visible(False)
        app.set_camera_view(position=[0, -7, 6], target=[0, 0, 0])
        material = app.create_pbr_material(roughness=0.8)
        app.scene.add_mesh(path="/receiver", mesh_data=ke.geometry.create_plane_data(12, ke.UpAxis.Z),
                           material=material)
        rig = app.scene.define_prim("/rig", ke.scene.PrimType.XFORM)
        rig.set_local_translation(ke.Vec3(0, 0, 3))

        def capture():
            for _ in range(32):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        for kind, value_type in (
            ("directional", ke.scene.DirectionalLight), ("point", ke.scene.PointLight),
            ("spot", ke.scene.SpotLight), ("rect", ke.scene.RectLight),
            ("dome", ke.scene.DomeLight),
        ):
            subject = app.scene.define_prim("/rig/light", ke.scene.PrimType.LIGHT)
            light = value_type()
            light.intensity = 0.3 if kind == "dome" else 20
            if kind == "directional":
                light.direction = ke.Vec3(0, 0, -1)
                light.ambient = ke.Vec3(0, 0, 0)
            getattr(subject, f"set_{kind}_light")(light)
            lit = capture()
            rig.set_visible(False)
            dark = capture()
            assert np.mean(lit - dark) > 0.5, f"{kind} did not illuminate the receiver: {lit.mean()=}, {dark.mean()=}"
            rig.set_visible(True)
            rig.set_active(False)
            app.scene.define_prim(f"/marker_{kind}", ke.scene.PrimType.XFORM)
            inactive = capture()
            assert np.abs(inactive - dark).mean() < 2, kind
            assert rig.is_visible()
            rig.set_active(True)
            restored = capture()
            assert np.abs(restored - lit).mean() < 4, kind
            app.scene.remove_prim("/rig/light")
            removed = capture()
            assert np.abs(removed - dark).mean() < 2, kind
            print(f"PASS: {args.renderer} {kind} creation, hierarchy state, and removal")
    finally:
        app.close()


if __name__ == "__main__":
    main()

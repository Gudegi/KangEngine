"""Render an external USD scene and verify floor-relative camera navigation."""

from pathlib import Path
import argparse
import importlib.util
import json
from unittest.mock import patch

import numpy as np
from PIL import Image

import kangengine as ke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/usd_scene_smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source = Path(__file__).resolve().parents[2] / "view_usd_scene.py"
    spec = importlib.util.spec_from_file_location("usd_viewer", source)
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    app = viewer.USDSceneViewer(args.usd_file.resolve(), 1.0, False, False)
    app.initialize(width=1280, height=720, hide_ui=True, up_axis=app.result.up_axis, headless=True)
    try:
        app.set_vsync(enabled=False)
        app.setup()
        assert len(app.mesh_views) == len(app.meshes) > 0
        camera = app.get_camera()
        start = viewer._array(camera.get_camera_pos())

        for _ in range(3):
            app.draw_frame()
        initial = args.output / "interior.png"
        assert app.write_pixels_png(path=str(initial), flip_y=True)
        pixels = np.asarray(Image.open(initial).convert("RGB"))
        assert pixels.std() > 5, "The scene rendered a nearly uniform image"

        # Exercise the same key handler as the interactive viewer, with no
        # dependence on physical keyboard focus in a hidden window.
        up_idx = 2 if app.result.up_axis == ke.UpAxis.Z else 1
        with patch.object(app, "is_key_pressed", side_effect=lambda key: key == ke.keys.W):
            app._move_camera(0.5)
        walked = viewer._array(camera.get_camera_pos())
        assert abs(walked[up_idx] - start[up_idx]) < 1e-5
        assert np.isclose(np.linalg.norm(walked - start), 0.75 * app.units_per_meter)
        with patch.object(app, "is_key_pressed", side_effect=lambda key: key == ke.keys.E):
            app._move_camera(0.2)
        raised = viewer._array(camera.get_camera_pos())
        assert np.isclose(raised[up_idx] - walked[up_idx], 0.3 * app.units_per_meter)
        for _ in range(3):
            app.draw_frame()
        moved = args.output / "after_movement.png"
        assert app.write_pixels_png(path=str(moved), flip_y=True)
        moved_pixels = np.asarray(Image.open(moved).convert("RGB"))
        assert np.mean(np.abs(pixels.astype(float) - moved_pixels)) > 1

        app._reset_camera()
        np.testing.assert_allclose(viewer._array(camera.get_camera_pos()), start)
        app.normal_maps_enabled = False
        app._apply_normal_map_toggle()
        app.normal_maps_enabled = True
        app._apply_normal_map_toggle()

        # A source image used as color and as data must not share GPU identity.
        texture_paths = [mesh.diffuse_texture_path for mesh in app.meshes if mesh.diffuse_texture_path]
        if texture_paths:
            color = app.load_texture(texture_paths[0], srgb=True)
            data = app.load_texture(texture_paths[0], srgb=False)
            assert color is not data
            assert app.load_texture(texture_paths[0], srgb=True) is color

        report = {
            "source": str(args.usd_file.resolve()),
            "meshes": len(app.meshes),
            "triangles": sum(mesh.index_count // 3 for mesh in app.meshes),
            "materials": len({mesh.material_path for mesh in app.meshes}),
            "textures": len(app.textures),
            "warnings": app.result.diagnostics.warnings,
            "lights": [
                {"path": light.prim_path, "type": str(light.type), "visible": light.visible}
                for light in app.result.lights
            ],
            "camera_exposure": app.result.cameras[0].linear_exposure_scale if app.result.cameras else 1.0,
            "camera_start": start.tolist(),
            "camera_after_walk": walked.tolist(),
            "camera_after_rise": raised.tolist(),
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()

"""Optional ovrtx probe consuming render_scene_input_export's CPU data.

Requires ovrtx 0.7.0.1794596436 and ovstage 0.4.0.1794520352.
Run with --input scene_input.json --output build/render-boundary/ovrtx/results.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ["NVIDIA_OMNIVERSE_PRODUCT_TELEMETRY_DISABLED"] = "1"

import ovrtx
import ovstage


BOOTSTRAP = '''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "World" {
    def Camera "Camera" {
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float focalLength = 24
        float horizontalAperture = 24
        float verticalAperture = 24
        float2 clippingRange = (0.1, 100)
    }
    def DistantLight "Light" {
        float inputs:intensity = 2500
        color3f inputs:color = (1, 1, 1)
    }
}
def Scope "Render" {
    def RenderProduct "Camera" {
        rel camera = </World/Camera>
        int2 resolution = (256, 256)
        token omni:rtx:rendermode = "PathTracing"
        int omni:rtx:pathtracing:spp = 4
        int omni:rtx:pathtracing:totalSpp = 16
        rel orderedVars = </Render/Camera/LdrColor>
        def RenderVar "LdrColor" {
            string sourceName = "LdrColor"
        }
    }
}
'''


def mesh_layer(mesh):
    points = ", ".join(str(tuple(p)) for p in mesh["points"])
    normals = ", ".join(str(tuple(n)) for n in mesh["normals"])
    counts = [3] * (len(mesh["indices"]) // 3)
    return f'''#usda 1.0
(defaultPrim = "Mesh")
def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
    point3f[] points = [{points}]
    normal3f[] normals = [{normals}] (interpolation = "vertex")
    int[] faceVertexCounts = {counts}
    int[] faceVertexIndices = {mesh["indices"]}
    uniform token subdivisionScheme = "none"
    rel material:binding = </Mesh/Material>
    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
    def Material "Material" {{
        token outputs:surface.connect = </Mesh/Material/Shader.outputs:surface>
        def Shader "Shader" {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.8, 0.15, 0.05)
            float inputs:metallic = 0
            float inputs:roughness = 0.7
            token outputs:surface
        }}
    }}
}}
'''


@contextmanager
def query_path(stage, path):
    with ovstage.PathDictionary(stage) as paths:
        path_list = paths.create_path_list_from_strings([path])
        try:
            with stage.query_from_path_list(path_list) as query:
                yield query
        finally:
            paths.destroy_path_list(path_list)


def render_image(renderer, ordinal):
    products = renderer.step(
        render_products={"/Render/Camera"}, delta_time=1 / 60, ordinal=ordinal
    )
    frame = products["/Render/Camera"].frames[-1]
    mapped = frame.render_vars["/Render/Camera/LdrColor"].map(device=ovrtx.Device.CPU)
    try:
        # The CPU copy survives unmap and the next renderer step.
        view = np.from_dlpack(mapped)
        pixels = view.copy()
        del view
    finally:
        mapped.unmap()
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload["schema_version"] != 1 or payload["mesh"]["colors"]:
        raise ValueError("This probe supports schema 1 without vertex colors")
    for frame in payload["frames"]:
        material = frame["material"]
        if (material["model"] != "PBR" or material["unlit"] or
                material["has_native_textures"] or material["base_color"][3] != 1):
            raise ValueError("This probe supports opaque, lit, untextured PBR")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderer = ovrtx.Renderer(config=ovrtx.RendererConfig(
        cache_root_path=str(output.parent / "runtime-cache"),
        log_file_path=str(output / "ovrtx.log"), keep_system_alive=False,
    ))
    stage = ovstage.Stage("kangengine.render-boundary")
    attached = False
    references = {}
    images = {}
    reconnects = 0
    try:
        renderer.attach_ovstage(stage)
        attached = True
        ovstage.population.open_usd_from_string(stage, BOOTSTRAP, ordinal=1)
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        layer = mesh_layer(payload["mesh"])

        for frame_idx, frame in enumerate(payload["frames"], start=1):
            ordinal = frame_idx * 2
            current_ids = {item["id"] for item in frame["instances"]}
            reconnect = bool(images) and bool(current_ids) and not (current_ids & references.keys())
            changed = False
            removed_ids = references.keys() - current_ids
            for instance_id in removed_ids:
                ovstage.population.remove_usd(stage, references.pop(instance_id))
                changed = True
            for instance_id in current_ids - references.keys():
                references[instance_id] = ovstage.population.add_usd_reference_from_string(
                    stage, layer, f"/World/Instance_{instance_id}"
                )
                changed = True
            if changed:
                ovstage.population.apply_usd_changes(stage, ordinal=ordinal)
            for instance_id in removed_ids:
                # A literal path-list query counts requested paths, not live prims.
                filter_ = ovstage.Filter([ovstage.Predicate(
                    "usd-path", ovstage.FilterOp.PREFIX, [f"/World/Instance_{instance_id}"]
                )])
                with stage.query(filter=filter_) as query:
                    assert query.result().total_prim_count == 0

            # Population and data writes use separate ordinals before publication.
            write_ordinal = ordinal + 1
            for instance in frame["instances"]:
                matrix = np.asarray(instance["world_matrix"], dtype=np.float64)
                # GLM column-vector math -> USD row-vector math.
                matrix = np.ascontiguousarray(matrix.T[None])
                with query_path(stage, f'/World/Instance_{instance["id"]}') as query:
                    stage.write_attribute(query, "omni:xform", ordinal=write_ordinal,
                                          tensors=matrix, is_array=False,
                                          semantic=ovstage.AttributeSemantic.MATRIX).wait()
                material = frame["material"]
                with query_path(stage, f'/World/Instance_{instance["id"]}/Material/Shader') as query:
                    for attribute, value in (
                        ("inputs:diffuseColor", [material["base_color"][:3]]),
                        ("inputs:metallic", [material["metallic"]]),
                        ("inputs:roughness", [material["roughness"]]),
                    ):
                        stage.write_attribute(query, attribute, ordinal=write_ordinal,
                                              tensors=np.asarray(value, dtype=np.float32),
                                              is_array=False).wait()
            stage.advance_write_floor(write_ordinal, ovstage.Scope.ALL).wait()
            if reconnect:
                # The pinned ovrtx build loses new meshes when no previous mesh survives.
                attached = False
                renderer.detach_ovstage()
                renderer.attach_ovstage(stage)
                attached = True
                reconnects += 1
            pixels = render_image(renderer, write_ordinal)
            images[frame["name"]] = pixels
            Image.fromarray(pixels).save(output / f'{frame["name"]}.png')
            print(frame["name"], pixels.shape, flush=True)

        background = images["removed"][..., :3].astype(np.int16)
        centers = {}
        checks = {"removed_image_empty": bool(np.max(background) <= 20)}
        for name in ("initial", "moved", "recolored", "recreated", "recreated_moved", "replaced"):
            difference = images[name][..., :3].astype(np.int16) - background
            mask = np.max(np.abs(difference), axis=2) > 20
            visible = np.count_nonzero(mask) >= 100
            checks[f"{name}_visible"] = bool(visible)
            centers[name] = float(np.nonzero(mask)[1].mean()) if visible else None
        checks["translation_applied"] = (
            centers["initial"] is not None and centers["moved"] is not None
            and centers["moved"] > centers["initial"] + 20
        )
        for name, reference in (("recreated", "initial"), ("recreated_moved", "moved"),
                                ("replaced", "initial")):
            checks[f"{name}_position"] = (
                centers[reference] is not None and centers[name] is not None
                and abs(centers[name] - centers[reference]) < 10
            )
        checks["material_changed"] = bool(
            np.abs(images["recolored"].astype(float) - images["moved"]).mean() > 1
        )
        report = {
            "build": str(ovrtx.get_build_info()), "checks": checks,
            "centers_x": centers, "frames": list(images),
            "output_shape": list(images["initial"].shape),
            "output_dtype": str(images["initial"].dtype),
            "output_color_space": "tone-mapped sRGB", "render_mode": "PathTracing",
            "stage_reconnects": reconnects,
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AssertionError(f"ovrtx probe failed: {', '.join(failed)}")
        print("PASS: ovrtx scene input lifecycle", flush=True)
    finally:
        if attached:
            renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == "__main__":
    main()

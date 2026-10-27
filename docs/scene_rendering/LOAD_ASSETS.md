# Load Assets

For simulated MJCF, URDF, and USD robots, see
[Articulations](../simulation/ARTICULATION.md).

Asset loaders parse files into data. Scene helpers or visual classes decide how
that data is displayed.

## OBJ with materials

```python
result: ke.ObjImportView = self.scene.add_obj(
    path="/environment",
    obj_path=obj_file,
    double_sided=True,
)
result.root.set_local_scale(scale=ke.Vec3(scale, scale, scale))
```

Run:

```bash
python ./python/examples/view_obj_scene.py --obj-file /path/to/model.obj
```

<details>
<summary>Complete source: <code>view_obj_scene.py</code></summary>

```{literalinclude} ../../../../python/examples/view_obj_scene.py
:language: python
:linenos:
```

</details>

## FBX mesh

```python
meshes: list[ke.asset.FBXMeshInfo] = ke.asset.FBXLoader.load_meshes(
    fbx_path=fbx_file,
    scale=scale,
)
for i, mesh in enumerate(meshes):
    self.scene.add_mesh(
        path=f"/fbx/mesh_{i}",
        mesh_data=mesh.mesh_data,
        material=material,
    )
```

## USD scenes

Build with USD support (`make build_usd_python`), then open the scene's root
layer. Referenced layers and textures must remain in their original relative
locations:

```bash
python python/examples/view_usd_scene.py /path/to/scene.usda
```

The viewer uses the stage's Y-up or Z-up axis and starts at its first authored
perspective camera. Use **Reset camera** to return there or **Frame scene** to
see the whole scene. Hold the right mouse button to look around; **WASD** moves
parallel to the floor, **Q/E** moves down/up, and **Shift** increases speed.
Movement is free of collision constraints. Physics and articulation loading
are separate from this viewer.

USD RectLight and DomeLight objects are created as scene lights, with the
source camera's exposure and ACES tone mapping. The **Exposure compensation**
control changes viewing exposure without modifying the source light values.
See [Scene Lighting](LIGHTING.md) for the supported controls and rendering limits.

The loader includes visible meshes with effective `default` or `render`
purpose, including native USD instances and material subsets. It respects
inherited visibility and purpose. A mesh with collision properties is still
displayed if it is also visible render geometry.

`UsdPreviewSurface` materials retain base color/opacity, metallic, roughness,
emission, normal maps and double-sided state. Texture connections can pass through
`NodeGraph` outputs. Supported `UsdUVTexture` maps include RGB colors, R-channel
scalar maps, RGB scalar maps (sampled from red), separate R/RGB opacity masks,
and packed G-roughness / B-metallic maps. Unsupported shader connections produce
diagnostics. Color maps use sRGB decoding unless `sourceColorSpace` is `raw`; normal and scalar
maps remain linear. For manually loaded textures, see
[Materials and Textures](MATERIALS_TEXTURES.md#load-and-attach-textures).

```python
result: ke.asset.USDImportResult = ke.asset.USDLoader.parse(
    usd_path=usd_file,
    scale=1.0,
)
meshes: list[ke.asset.USDMeshInfo] = result.meshes
```

Mesh vertices, bounds and camera positions use authored scene units multiplied
by `scale`; axes are not rotated. `result.meters_per_unit` reports the original
stage unit. Camera vertical FOV is in degrees.

The returned meshes have transforms baked into their vertices. The result is
for static display, not editing and saving the original USD scene; keep the
source USD for that purpose. `prim_path` and `material_path` identify the source
objects. The viewer uses fallback lighting when no supported USD lights were
imported. MDL/MaterialX networks, arbitrary texture transforms and camera roll
are not reproduced.

## Examples

- `python/examples/view_fbx_mesh.py`
- `python/examples/view_obj_scene.py`
- `python/examples/view_usd_scene.py`
- `python/examples/usd_file_bridge.py`

![Imported asset with materials](../images/scene_rendering/load_assets.png)

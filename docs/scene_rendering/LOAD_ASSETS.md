# Load Assets

For simulated MJCF, URDF, and USD robots, see
[Articulations](../simulation/ARTICULATION.md).

Asset loaders parse files into data. Scene helpers or visual classes decide how
that data is displayed.

## OBJ with materials

The OBJ and FBX snippets run inside `App.setup()`, with
`import kangengine as ke`. Set `obj_file` or `fbx_file` to your asset path
and `scale` to its unit conversion factor (use `1.0` for meter-scale assets).
The FBX snippet also uses a material created through the App; see
[Materials and Textures](MATERIALS_TEXTURES.md).

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

## PLY meshes and point clouds

`asset.load_ply()` returns `scene.MeshData` without an App or graphics context:

```python
import kangengine as ke

mesh: ke.scene.MeshData = ke.asset.load_ply(path="model.ply")
```

PLY 1.0 ASCII and little-/big-endian binary files are supported. Coordinates and
scale are preserved. The loader reads positions, normals, RGB, and vertex UVs
(`u/v`, `s/t`, or `texture_u/texture_v`). Separate `multi_texture_vertex` /
`multi_texture_face` UV tables support one texture and use the OBJ UV convention
after loading. Multiple texture IDs are rejected.

Triangulate concave polygons before loading; convex faces are triangulated
automatically. Missing mesh normals are generated, and missing UVs become zero.
Point clouds have empty `indices` and keep missing attributes empty.

Integer RGB is normalized to `[0, 1]`; floating RGB must already be in that range.
Colors are linear by default; pass `colors_srgb=True` for sRGB input. Alpha and
unknown properties are ignored. Materials and image references are not loaded.

Inside `App.setup()`, add a mesh:

```python
material: ke.material.ColorMaterial = self.create_color_material()
material.unlit = True
self.scene.add_mesh(
    path="/ply", mesh_data=mesh, material=material, color=ke.Vec4(1, 1, 1, 1),
)
```

For point clouds, add alpha to the RGB values for the point drawing API:

```python
import numpy as np

points = np.asarray(mesh.vertices, dtype=np.float32)
rgb = mesh.colors
colors = None
if rgb:
    colors = np.ones((len(points), 4), dtype=np.float32)
    colors[:, :3] = np.asarray(rgb)
self.debug.overlay.points(path="/ply", points=points, colors=colors, size=6.0)
```

The viewer selects mesh or point rendering automatically. `--up-axis Y|Z`
sets the viewer's up axis (default `Y`), without rotating the source coordinates:

```bash
python python/examples/asset/view_ply.py --ply-path /path/to/model.ply
```

For a mesh with UVs, use `--texture-path` to supply an image. The viewer loads it
as sRGB and displays it without lighting; image paths are not detected automatically:

```bash
python python/examples/asset/view_ply.py \
  --ply-path "/path/to/model.ply" \
  --texture-path "/path/to/model_color.jpg" --up-axis Z
```

## Rigid body sources

These loaders prepare descriptions for the [Rigid Bodies](../simulation/RIGID_BODIES.md)
workflow. Construct the source first, then register or render instances.

### OBJ files

This optional example needs your own meter-scale, Z-up `chair.obj`. Keep the
OBJ, MTL, and referenced textures together:

```python
import kangengine as ke

obj_asset: ke.asset.RigidDesc = ke.asset.load_obj_rigid(obj_path="chair.obj")
```

Coordinates and scale are preserved. Material subsets, normals, and UVs are
retained. The default triangle-mesh collider supports static or kinematic bodies.
Dynamic triangle-mesh collision is unsupported by this rigid-body API.

For a box-shaped OBJ, supply a primitive collider in the same local frame:

```python
import kangengine as ke

box_collider: ke.asset.CollisionGeomDesc = ke.asset.CollisionGeomDesc(
    type=ke.asset.CollisionGeomDescType.BOX,
    size=[0.5, 0.3, 0.4],  # Half extents in meters; match your mesh.
    position=[0.0, 0.0, 0.0],
    quat_xyzw=[0.0, 0.0, 0.0, 1.0],
)
textured_box: ke.asset.RigidDesc = ke.asset.load_obj_rigid(
    obj_path="textured_box.obj", collision_geoms=[box_collider]
)
```

For a concave object such as a cup, supply several primitive or convex pieces in
the same local frame as its visual mesh. The OBJ loader does not perform
automatic convex decomposition. Prepare the pieces separately and supply them
through `collision_geoms`.

### Single-body imports

As an alternative to building a description, pass a single-body MJCF returned
by `world.load_mjcf(mjcf_path=...)` to `world.add_rigid()`.
`ke.asset.RigidDesc.from_articulation(description)` is available when explicit
conversion from a single-body `ArticulationDesc` is needed.

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

OmniPBR materials retain constant diffuse color, metallic, and roughness values,
plus a directly assigned `diffuse_texture` and its `diffuse_tint`. The texture
replaces the constant color; `colorSpace = "raw"` disables sRGB decoding.
Other OmniPBR maps, albedo adjustments, and connected MDL networks are unsupported.

Articulation imports retain these mesh-bound materials and their base-color,
normal, metallic, roughness, packed metallic-roughness, and emissive maps.
Textures are loaded by the App when visuals are attached, preserving color-space
settings and sharing cached textures across instances. Articulation opacity maps,
alpha cutout, and per-face material assignments are not imported.

The parsing snippet uses `usd_file`, the path to your root USD layer:

```python
import kangengine as ke

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

# Load Assets

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
meshes: list[ke.asset.FBXStaticMeshInfo] = ke.asset.FBXLoader.load_meshes(
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

## USD

USD is available only in an explicitly USD-enabled development build. The
distributed wheel configuration intentionally disables it.

Examples:

- `python/examples/view_fbx_mesh.py`
- `python/examples/view_obj_scene.py`
- `python/examples/view_usd_scene.py`
- `python/examples/usd_file_bridge.py`

![Imported asset with materials](../images/scene_rendering/load_assets.png)

# Scene and Rendering

Use the scene layer for inspectable paths, transforms, materials, imported
assets, and interactive tools.

- [Materials and Textures](MATERIALS_TEXTURES.md)
- [Load Assets](LOAD_ASSETS.md)
- [Camera and Viewer](CAMERA_VIEWER.md)
- [Video Recording](VIDEO_RECORDING.md)
- [Debug Visualization and Text Rendering](DEBUG_VISUALIZATION.md)

Most application code should use `app.scene` and prim-backed views. Use
`ke.render` directly only for textures, buffers, or renderer experiments.
Custom graphics pipelines are exposed through the C++ and Python renderer
hook APIs. Python-authored definitions created with
`App.create_scene_hook_pipeline()` appear in the Resource panel.

Primitive mesh factories live under `ke.geometry`:

- `create_plane_data()`
- `create_cube_data()` and `create_box_data()`
- `create_sphere_data()`
- `create_capsule_data()` and `create_cylinder_data()`
- `create_cone_data()` and `create_arrow_data()`

These functions return `MeshData`; pass the result to
`app.scene.add_mesh(...)` to create a renderable Prim. See
[First Scene](../getting_started/FIRST_SCENE.md) for the complete workflow and
the `ke.geometry` API Reference for exact signatures.

# Materials and Textures

For common scenes, start with the standard material bundle:

```python
self.standard_materials = self.create_standard_materials()
self.add_ground()
```

Python material facades live under `ke.material`; low-level shader and texture
objects live under `ke.render`. Scene and visualization APIs accept any
material subtype through the same `material=` argument, so Phong and PBR are
interchangeable at the binding point:

```python
material = self.create_phong_material(diffuse=ke.vec3(0.8, 0.3, 0.1))
# Or: material = self.create_pbr_material(
#     base_color=ke.vec4(0.8, 0.3, 0.1, 1.0), roughness=0.6
# )

mesh = self.scene.add_mesh("/mesh", mesh_data, material)
mesh.set_casts_shadow(True)
mesh.set_double_sided(False)
```

Phong and PBR parameters are intentionally different; replacing the material
does not translate `diffuse`/`shininess` into `base_color`/`roughness`.
`create_phong_material()` and `create_pbr_material()` return independent,
mutable instances. The entries in `standard_materials` are shared defaults.

For imported OBJ/MTL scenes, `SceneContext.add_obj(...)` creates material
subsets and binds diffuse, specular, alpha, and normal textures when available.

Examples:

- `python/examples/test_phong_texture.py`
- `python/examples/view_obj_scene.py`
- `python/examples/view_pbr.py`
- `python/examples/view_pbr_presets.py`

![PBR presets](../images/scene_rendering/materials_and_textures.png)

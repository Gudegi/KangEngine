# Scene Lighting

Create lights as prims under `/lights`, then use the prim to position, rotate,
scale or hide them. Changes appear on the next rendered frame.

Examples assume `import kangengine as ke` and run in `App.setup()`, using
`self.scene`. The manual examples use **Y-up**, with one scene unit treated as
one meter. The USD example keeps the imported stage's axes and units.

## Choose a light

| Type | Typical use | Direction | Visible light limit | Shadows | Materials |
| --- | --- | --- | --- | --- | --- |
| Directional | Sun | `direction` follows emitted rays | 1 | Yes | common, Phong, PBR and ground |
| Point | Bulb | All directions | 4 | No | PBR and PBR ground |
| Spot | Flashlight or stage light | `direction` follows emitted rays | 2 | No | PBR and PBR ground |
| Rect | Ceiling panel | Emits along local -Z | 16 | No | PBR and PBR ground |
| Dome | Uniform fill light | No direction | No limit (additive) | No | common, Phong, PBR and ground |

Point, Spot and Rect do not illuminate `common` or Phong materials. For them,
use a [PBR material](MATERIALS_TEXTURES.md#create-a-pbr-material).
Dome currently supplies uniform ambient illumination, not an HDRI background
or environment reflections.

Directional, Point and Spot use the first eligible lights encountered under
`/lights`; excess lights are ignored. Lights are selected depth-first through
the scene hierarchy, in creation order within each parent, without sorting by
distance or brightness. More than 16 eligible Rect lights raises an error. Hidden or
inactive lights, including those under hidden or inactive parents, are excluded.

## Common rules

Use **Add → Light** and the inspector for interactive creation and editing.
For Python, use `define_prim()` to create a light and `get_prim_at_path()` to
look up an existing one. An existing prim at the requested path is reused by
`define_prim()`.

### Coordinates and updates

| Type | Setter input and parent transforms | Getter result |
| --- | --- | --- |
| Directional | Local direction follows the prim transform; translation has no effect | Local direction |
| Point | `position` sets the prim's local translation; `range` stays in world scene units and does not scale | Parent-relative position |
| Spot | Same position/range rules as Point; local direction follows the prim transform | Parent-relative position and local direction |
| Rect | Local dimensions and local -Z follow the prim transform, including parent scale | Original dimensions and emission values |
| Dome | Translation, rotation and scale have no effect on uniform illumination | Original emission values |

All `get_*_light()` methods return copies of local values. Read a light, edit
its properties and pass it back to the matching setter; no world-to-local
conversion is needed, even under transformed parents. Directional and Spot
`direction` both point along emitted rays. Point and Spot `position` reflects
the prim's current local translation. Setting an older copy can overwrite a
position edit made since that copy was read.

For a world-space position, use the prim's `get_world_translation()` rather
than the light getter.

For the examples below, use these substitutions in a Z-up scene:

| Setting | Y-up examples | Z-up equivalent |
| --- | --- | --- |
| Position at height `h` | `(0, h, 0)` | `(0, 0, h)` |
| Downward Directional | `(0, -1, 0)` | `(0, 0, -1)` |
| Downward Spot | `(0, -1, 0)` | `(0, 0, -1)` |
| Downward Rect | Rotate -90 degrees about local X | Identity rotation |

When updating older code that used a source-facing Directional vector, negate
that vector once, including values passed to `App.set_light_direction()`.
Spot and Rect directions are unchanged. Remove world-to-local conversions
previously applied to light getter results; those results are now local.

### Brightness

`color` is a linear RGB multiplier. Directional, Point and Spot use an
intensity multiplier; their values are not interchangeable with USD light
intensity. Rect and Dome additionally expose exposure and color temperature.

The manual examples use a viewing exposure of 1:

```python
self.set_tone_map(mode=ke.render.ToneMapMode.ACES_NARKOWICZ, exposure=1.0)
```

The example intensities are starting values for these dimensions and distances,
not universal presets. For a manually authored scene, fix viewing exposure,
place the light, then adjust its intensity. For imported USD, start with the
source light values and camera exposure together.

## Directional lights

Use a directional light for sunlight or other illumination without distance
falloff. `App` creates one at `/lights/default_directional`; update that prim
to change the default light:

```python
sun: ke.scene.Prim = self.scene.get_prim_at_path(path="/lights/default_directional")
light: ke.scene.DirectionalLight = ke.scene.DirectionalLight()
light.direction = ke.Vec3(0.0, -1.0, 0.0)
light.intensity = 2.0
light.ambient = ke.Vec3(0.1, 0.1, 0.1)
sun.set_directional_light(light=light)
```

`direction` points **from the light toward the scene**. With an unrotated prim, the
example lights upward-facing surfaces in a Y-up scene. Rotating the prim or
its parents rotates this direction; translating the prim has no effect.
`color` is a linear RGB multiplier and `intensity` scales the direct light.
`ambient` adds uniform illumination independently of `intensity`.
Use it for a simple fill alongside the default sun. Use a Dome when the fill
should be controlled as a separate scene object.

Only one visible directional light is used. Update the default rather than
adding a second one. For a scene lit only by point, spot, rectangle or dome
lights, hide the default with `sun.set_visible(visible=False)`.

## Point lights

Use a point light for illumination spreading in all directions from a position:

```python
point: ke.scene.Prim = self.scene.define_prim(
    path="/lights/point", prim_type=ke.scene.PrimType.LIGHT,
)
light: ke.scene.PointLight = ke.scene.PointLight()
light.position = ke.Vec3(0.0, 2.0, 0.0)
light.intensity = 5.0
light.range = 8.0
point.set_point_light(light=light)
```

`position` sets the prim's local translation relative to its parent. You can
then move it with `set_local_translation()`. Light fades with distance and
reaches zero at `range`, measured in world scene units. Scaling the prim does
not change `range`. `color` and `intensity` control the light's color and
strength.

To change intensity without moving the point light:

```python
light: ke.scene.PointLight = point.get_point_light()
light.intensity = 3.0
point.set_point_light(light=light)
```

## Spot lights

Use a spot light to illuminate a cone. This example points down in a Y-up scene:

```python
import math

spot: ke.scene.Prim = self.scene.define_prim(
    path="/lights/spot", prim_type=ke.scene.PrimType.LIGHT,
)
light: ke.scene.SpotLight = ke.scene.SpotLight()
light.position = ke.Vec3(0.0, 3.0, 0.0)
light.direction = ke.Vec3(0.0, -1.0, 0.0)
light.intensity = 10.0
light.range = 8.0
light.inner_cone_angle = math.radians(20.0)
light.outer_cone_angle = math.radians(30.0)
spot.set_spot_light(light=light)
```

`direction` points **from the light toward its target**, in the prim's local
frame. The prim and its parents rotate that direction. Position, color,
intensity and range behave like a point light.

Cone angles are half-angles from the center direction, in radians. Set
`inner_cone_angle < outer_cone_angle`: light is strongest inside the inner
cone and fades to zero at the outer cone.

## Rectangle lights

A rectangle is centered in its local XY plane and emits toward local **-Z**.
For example, rotate this panel to shine down in a Y-up scene:

```python
import math

panel: ke.scene.Prim = self.scene.define_prim(
    path="/lights/ceiling", prim_type=ke.scene.PrimType.LIGHT,
)
light: ke.scene.RectLight = ke.scene.RectLight()
light.width = 2.0
light.height = 1.0
light.intensity = 5.0
light.enable_color_temperature = True
light.color_temperature = 4200.0
panel.set_rect_light(light=light)
panel.set_local_translation(translation=ke.Vec3(0.0, 2.5, 0.0))
panel.set_local_rotation_axis_angle(
    axis=ke.Vec3(1.0, 0.0, 0.0), angle_radians=-math.pi / 2,
)
```

Dimensions use local scene units. `get_rect_light()` returns the authored
dimensions and emission values; it does not bake in the parent transform.
Edit a returned value and call `set_rect_light()` to apply it. Position and
orientation remain on the prim.

- `color` is a linear RGB multiplier.
- `intensity` follows USD's luminance convention before exposure.
- `exposure` is in stops: +1 doubles emission, -1 halves it.
- `normalize=False` keeps luminance constant, so a larger rectangle emits
  more total light. With `normalize=True`, emission is divided by the rectangle's
  world-space area, including parent scale.
- `color_temperature` uses kelvin when `enable_color_temperature` is true.
- `diffuse` and `specular` scale the corresponding lighting contributions.

Sharp specular highlights may show artifacts. Light textures and emission
shaping are not supported. Objects between a panel and a surface do not block
its light.

## Dome lights

Use a dome light to add uniform diffuse environment illumination:

```python
sky: ke.scene.Prim = self.scene.define_prim(
    path="/lights/sky", prim_type=ke.scene.PrimType.LIGHT,
)
light: ke.scene.DomeLight = ke.scene.DomeLight()
light.intensity = 0.2
sky.set_dome_light(light=light)
```

Color, intensity, exposure and color temperature work as described above.
`diffuse` scales the illumination; `specular` and `normalize` have no effect.
All Dome contributions add to the directional light's `ambient`. Set `ambient`
to zero if only the Domes should provide fill.
Objects do not block dome illumination. Environment reflections and textured
domes are not supported.

## USD scenes and camera exposure

`USDLoader.parse()` returns rectangle and dome lights in `result.lights`.
Each entry retains its source path, evaluated world transform, effective
visibility and authored light values. Other USD light types and unsupported
light textures/shaping produce diagnostics.

The [USD scene viewer](LOAD_ASSETS.md#usd-scenes) uses imported lights when
available and applies the first authored perspective camera's exposure with
ACES tone mapping.
Use **Exposure compensation (stops)** to adjust the view without changing the
authored light values. **Reset camera** restores the source camera's exposure.
Scenes without a camera use an exposure scale of 1.

Warnings appear on standard error and are also available in
`result.diagnostics.warnings` for your own UI or logging.

<details>
<summary>Apply USD lights and exposure in a custom viewer</summary>

For a custom viewer, [parse the USD](LOAD_ASSETS.md#usd-scenes) before calling
`App.initialize()`, and initialize with `up_axis=result.up_axis`. After applying
the imported mesh materials and selecting the first imported camera, use this
in `setup()` to add the lights and exposure. Here `result` is that
`ke.asset.USDImportResult`; use this instead of the manual light examples above.

```python
if result.lights:
    default_light: ke.scene.Prim = self.scene.get_prim_at_path(
        path="/lights/default_directional",
    )
    default_light.set_visible(visible=False)

    for source in result.lights:
        prim: ke.scene.Prim = self.scene.define_prim(
            path="/lights/usd" + source.prim_path,
            prim_type=ke.scene.PrimType.LIGHT,
        )
        if source.type == ke.scene.LightType.RECT:
            prim.set_rect_light(light=source.rect)
        elif source.type == ke.scene.LightType.DOME:
            prim.set_dome_light(light=source.dome)
        prim.set_world_matrix(matrix=source.world_transform)
        prim.set_visible(visible=source.visible)

    exposure = result.cameras[0].linear_exposure_scale if result.cameras else 1.0
    self.set_tone_map(mode=ke.render.ToneMapMode.ACES_NARKOWICZ, exposure=exposure)
```

Without supported imported lights, this example retains the default lighting.

</details>

Matching light and exposure values does not guarantee identical images across
renderers: tone mapping, shadows and indirect lighting also affect the result.

## Shadows

Directional lights support shadows. For a mesh created with
`self.scene.add_mesh()`, use `mesh.set_casts_shadow(enabled=True)` to include it
as a shadow caster. See [Mesh rendering options](MATERIALS_TEXTURES.md#mesh-rendering-options).
Point, Spot and Rect shadows are not supported, even when a mesh has shadow
casting enabled. Directional ambient and Dome fill remain unshadowed, so high
fill values can make directional shadows look faint.

## If the scene is too dark or too bright

- Check the material first: Point, Spot and Rect require PBR or PBR ground.
- Check that the light is under `/lights` and that neither it nor a parent is
  hidden or inactive. Check the count limits before adding more lights.
- Check directions against the table, including the scene's up-axis. Rect
  emits from only one side. Check Point/Spot `range` against the scene units.
- Check the default directional light. It may be hidden, or it may still be
  adding light alongside imported lights.
- For USD scenes, use the source camera exposure before changing source light
  intensity. Try the viewer's exposure compensation; +1 stop doubles viewing
  exposure and -1 halves it. **Reset camera** restores the source exposure.
- Missing indirect light, reflections and local-light shadows can produce a
  different result from another renderer even with matching input values.

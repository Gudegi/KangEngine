# Simulation Visualization

Display simulated objects and keep their visuals aligned with the latest
physics state. `ke.visual.sim.SimWorldVisualizer` connects a `KangSimWorld`
to an App and synchronizes poses before rendering.

For objects driven by supplied poses without physics, use
[`ke.visual.RigidVisual`](RIGID_BODIES.md#render-without-physics) and call
`apply_pose()` directly.

## Batched rendering

Use `add()` for normal simulation rendering, including batched environments.
The snippets extend an `App` subclass with `import kangengine as ke`, as in
[First Simulation](../getting_started/FIRST_SIMULATION.md). In `setup()`, create
`self.world` and register an articulation as shown in
[Articulations](ARTICULATION.md); keep the returned `SimArticulation` as
`record`. The following code also belongs in `setup()` and uses that record's
registered visual geometry and materials:

```python
self.visual: ke.visual.sim.SimWorldVisualizer = (
    ke.visual.sim.SimWorldVisualizer(app=self, world=self.world)
)
self.robot_visual: ke.visual.sim.VisualBatch = self.visual.add(
    sim_handle=record,
    path="/robot",
)
```

Add these methods to the same App class:

```python
def fixed_update(self, fixed_dt):
    self.world.advance(duration=fixed_dt)

def pre_render(self):
    self.visual.sync()
```

`add()` returns a `VisualBatch`, not an `ArticulationVisual`. It is the normal
rendering handle for both rigid bodies and articulations and uses an
ExternalBuffer-backed path.

Rigid and articulation batches choose their CPU or CUDA rendering path from the
simulation device. CUDA batches update render transforms directly from GPU
simulation state. Static rigid bodies use the CPU path in either kind of world.
Both paths preserve each visual part's local position, rotation, and scale.

Rigid bodies registered from a `RigidDesc` already carry their visual geometry
and material subsets. Use `visualizer.add(sim_handle=rigids, path="/props")`
without an MJCF path. The same call renders rigid descriptions using
their collision shapes when no visual parts are authored. Articulations can also
use the `VisualGeomDesc` parts in their registered `ArticulationDesc` without
reloading an MJCF. See [Rigid Bodies](RIGID_BODIES.md#prepare-a-rigid-description)
for textured objects whose visual surface differs from their colliders.

`VisualGeomDesc.material` accepts the same PBR and Phong materials used by the
renderer. A source material's `texture_paths` can defer texture loading until
the visual is attached to an App. Rigid and articulation visuals retain the
source appearance while creating their own App-owned runtime materials.

Simulation state and scene/render state are separate. Advance physics in
`fixed_update()` and synchronize visuals in `pre_render()` so rendering speed
does not alter the physics rate.

## Inspectable SceneGraph visuals

Use `add_scene_graph()` when individual prims must be visible and selectable in
the editor. In `setup()`, replace the `self.visual.add(...)` call above with:

```python
robot_visual: ke.visual.sim.VisualArticulationSceneGraph = self.visual.add_scene_graph(
    sim_handle=record,
    env_id=0,
)
robot_visual.set_color(color=(0.8, 0.8, 0.9, 1.0))
robot_visual.set_collision_visible(visible=True)
```

For an articulation, `robot_visual` is a `VisualArticulationSceneGraph` and
`robot_visual.articulation_visual` is the underlying `ArticulationVisual`
synchronized from the simulation. Most applications should use the record
rather than call the underlying bridge directly.

`VisualBatch.release()` removes one batch. `SimWorldVisualizer.release()`
releases every visual registered through that visualizer.

Use one representation deliberately:

- `add()` / `VisualBatch` for throughput and batched environments.
- `add_scene_graph()` for inspectable prims, selection, and collision display.

## Physics inspection

Open the viewport's eye icon → **Physics Inspector** to inspect colliders,
contacts, centers of mass, and coordinate axes. These displays work with Native
and OVRTX and are off by default.

`SimWorldVisualizer` connects the simulation automatically unless another source
is already connected. For a standalone `PhysicsWorld`, connect it after App
initialization; the menu is disabled until connected:

```python
self.physics_inspector.set_source(
    source=ke.physics.PhysXInspectionSource(world=self.physics_world),
)
```

Finish fetching physics results in `pre_render()` before the displays update.

```bash
python python/examples/rendering/physics_inspector.py --renderer native
python python/examples/rendering/physics_inspector.py --renderer ovrtx
```

### Colliders and selection

Enable **Static Colliders** (pink) or **Dynamic Colliders** (green) to see the
actual collision shapes. Select an object to highlight its colliders in yellow
and inspect its body type, shape dimensions, and local pose.

`SimWorldVisualizer` connects selection automatically. For a visual created with
`scene.add_mesh()`, connect its physics body yourself:

```python
self.physics_inspector.bind_visual(visual, body)
```

For an ExternalBuffer visual, also pass `instance_index` to identify the instance.

Enable **Pick Colliders** in Inspect mode to select visible collider categories
directly, even without visual meshes. In GPU worlds, this works only for static
colliders; select the visual mesh to inspect a dynamic body.

Boxes, spheres, capsules, convex meshes, triangle meshes, and heightfields are
supported. Infinite planes use a finite preview (`plane_size=10.0` scene units).
CPU collider lines remain visible through meshes; GPU dynamic colliders can be
hidden behind them.

Use **Refresh** after adding bodies, editing collision shapes, or moving/removing
static bodies. CPU dynamic poses and removals update automatically.
For GPU physics, initialize the GPU system before connecting; without a visualizer,
pass `gpu_system` to `PhysXInspectionSource`. Adding or removing GPU bodies requires
GPU system reinitialization, then **Refresh**.

Use `self.physics_inspector.set_source(None)` to disconnect. Changing the source
clears display toggles and body bindings; App shutdown cleans up automatically.

### Contacts

Under **Contacts**, enable **Normal**, **Force**, or **Contact Mode**.
The scale and thickness controls change the display, not the simulation.

- **Normal:** the reported contact normal direction.
- **Force:** average normal force over the last physics step, excluding friction.
- **Contact Mode:** estimated unloaded (gray), sticking (cyan), or sliding (orange)
  state based on force and relative motion, not the solver's friction state.

Contact markers can be hidden behind meshes. Separated points with negligible
force are hidden, but remain in the reported count. Sleeping CPU bodies may stop
reporting contacts.

CPU display requires `enable_contact_reports=True` (the default); GPU display
requires contact buffers in `PhysicsGpuSystem`. GPU inspection can add synchronization
cost while enabled. The menu warns if contacts exceed the buffer capacity.

### Centers of mass

Enable **Body / link COM** for yellow center-of-mass markers, or **Whole robot COM**
for each articulation's mass-weighted center in purple. The robot center includes
fixed-base links; static colliders are excluded. Markers stay the same size on
screen and remain visible through meshes.

Use **Refresh** after adding bodies or changing masses or local COM offsets.
GPU COM and coordinate-axis displays read positions back to CPU while enabled.

### Coordinate axes

**Body / link axes** and **Joint axes** show X/Y/Z in red/green/blue through
meshes. Body axes use the link origin, not its center of mass. Joint axes use the
joint frame attached to the child link. For a single-axis PhysX articulation joint,
red X marks the rotation or translation axis, so it may differ from the body's X axis.

## Render output inspection

Open the eye icon's **Render Output** menu to choose RGB, Depth,
Normals, or Object IDs. Unsupported modes are disabled.

| Output | Meaning |
| --- | --- |
| RGB | Normal rendering; the default |
| Depth | Depth along the camera's forward axis, in scene units. Black is background; white is the depth limit or farther. |
| Normals | World-space directions: XYZ from `[-1, 1]` becomes RGB `[0, 1]`. Native shows face normals; OVRTX shows surface normals. |
| Object IDs | Colors distinguish instances, not object categories or persistent IDs. OVRTX may color material subsets separately. |

Depth is not distance along the viewing ray. These views hide debug overlays and
selection outlines; picking still works. To select a view from Python:

```python
renderer: ke.render.Renderer = self.get_renderer()
renderer.set_view_mode(mode=ke.render.RenderViewMode.DEPTH, depth_max=10.0)
```

## Visual types

| Type | Role |
|---|---|
| `SimWorldVisualizer` | Registers simulation objects and synchronizes their visual state |
| `VisualBatch` | CPU/GPU external-buffer visual returned by `add()` |
| `VisualArticulationSceneGraph` | Inspectable per-link articulation record returned by `add_scene_graph()` |
| `VisualRigidSceneGraph` | Inspectable rigid-body record returned by `add_scene_graph()` |
| `VisualBodyPick` | Maps a renderer selection to environment, object, and body IDs |

## Examples

- `python/examples/sim_world_minimal.py`
- `python/examples/sim_world_multi_env.py`
- `python/examples/mjcf_dof_control.py`

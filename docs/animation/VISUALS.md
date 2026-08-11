# Pose Visualization

Choose a visual type by how its displayed pose is updated.

## How pose reaches geometry

### `ArticulationVisual`

**Simulation synchronization**

Body transforms are synchronized from a simulation articulation through
`ke.visual.sim.SimWorldVisualizer`. See
[Simulation Visualization](../simulation/VISUALIZATION.md) for registration
and synchronization.

### `ArticulatedSurface`

**Explicit `SkeletonState` application to rigid links**

`apply_state()` evaluates the supplied pose and writes transforms to rigid
link prims. It is not synchronized with a physics simulation.

### `SkinnedSurface`

**Explicit `SkeletonState` application to skinned vertices**

`apply_state()` converts the supplied pose to skinning matrices. The renderer
uses those matrices for GPU linear blend skinning.

```text
simulation articulation ──sync──────────────> ArticulationVisual
SkeletonState ──rigid-link transforms───────> ArticulatedSurface
SkeletonState ──skinning matrices / GPU LBS─> SkinnedSurface
```

## Visual types

| Type | Update mechanism | Typical use |
|---|---|---|
| `ArticulationVisual` | Simulation synchronization | PhysX robots and editor inspection |
| `ArticulatedSurface` | Explicit `apply_state()` to rigid links | Robot poses, comparison, and trails |
| `SkinnedSurface` | Explicit `apply_state()` to a skinned mesh | FBX, SMPL-family, and custom characters |
| `SkinVisual` | Integrated FBX skinning path | FBX loading and playback |
| `SkeletalVisual` | Explicit skeleton pose or motion sampling | Skeleton and motion inspection |

## Which FBX visual should I use?

Use `SkinVisual` for the shortest integrated FBX playback path. It loads the
motion and meshes together:

```python
character: ke.visual.SkinVisual = ke.visual.SkinVisual.from_fbx(
    app=self,
    material=material,
    fbx_path=fbx_path,
    path="/character",
)
character.apply_time(time=time, loop=True)
motion: ke.animation.SkeletonMotion = character.motion()
```

Use `SkinnedSurface` when FBX must follow the same surface API as SMPL or custom
characters, or when instances, motion trails, bind-geometry updates, or direct
`SkeletonState` application are needed:

```python
result: ke.asset.FBXImportResult = ke.asset.FBXLoader.parse(fbx_path=fbx_path)
surface: ke.visual.SkinnedSurface = (
    ke.visual.SkinnedSurface.create_from_fbx_result(
        app=self,
        path="/character",
        result=result,
    )
)
state: ke.animation.SkeletonState = result.motion.sample(time=time)
surface.apply_state(state=state)
```

`SkinVisual` is a higher-level integrated FBX bridge. It is not an internal
part of `SkinnedSurface`, and neither type requires the other.

## Skinned surfaces and SMPL

Imported FBX materials become retained PBR materials. Pass `material=...` to
override all parts with one material.

SMPL uses the same surface with an optional vertex-correction step before GPU
skinning:

```text
FBX:  fixed bind geometry ───────────────────────────────> GPU LBS
SMPL: shaped template + optional pose-dependent offsets ─> GPU LBS
```

The pose entry point remains `surface.apply_state()`. Pose correctives are an
optional geometry update immediately before it:

```python
model: ke.asset.SMPLXModel = ke.asset.SMPLXModel.load(path=model_path)
body: ke.asset.SMPLXBody = model.create_body(betas=betas)
surface: ke.visual.SkinnedSurface = body.create_visual(
    app=self,
    path="/smplx",
    material=material,
)

state: ke.animation.SkeletonState = motion.sample(time=time)
body.update_pose_correctives(
    surface=surface,
    state=state,
    enabled=pose_correctives,
)
surface.apply_state(state=state)
```

Without pose correctives, call only `surface.apply_state(state)`.

## Rigid-link surfaces

`ArticulatedSurface` loads an MJCF hierarchy without connecting it to a
simulation:

```python
robot: ke.visual.ArticulatedSurface = (
    ke.visual.ArticulatedSurface.create_from_mjcf(
        app=self,
        path="/robot",
        mjcf_path=mjcf_path,
        material=material,
    )
)
robot.apply_state(state=state)
```

Both surface types use the same explicit-pose pattern:

```python
surface.skeleton_tree
surface.apply_state(state=state)
surface.apply_pose(
    root_translation=root_translation,
    local_rotations_wxyz=local_rotations_wxyz,
)
```

Here `state` is `ke.animation.SkeletonState`. `SkeletonMotion.sample()` returns
a state. A standalone pose can be constructed directly:

```python
state: ke.animation.SkeletonState = (
    ke.animation.SkeletonState.from_rotation_and_root_translation(
        tree=surface.skeleton_tree,
        rotations_wxyz=local_rotations_wxyz,  # (num_joints, 4), WXYZ
        root_translation=root_translation,    # (3,)
        is_local=True,
    )
)
surface.apply_state(state=state)
```

Use `apply_pose(root_translation=..., local_rotations_wxyz=...)` when an
intermediate `SkeletonState` is not otherwise needed.

## Instances and motion trails

`SkinnedSurface` and `ArticulatedSurface` create independently posed instances
while sharing mesh, texture, and material assets. Skinned surfaces also share
skin weights and bind data:

```python
ghosts: list[ke.visual.SkinnedSurface | ke.visual.ArticulatedSurface] = [
    surface.create_instance(
        path=f"/character/ghost_{index}",
        color=(0.3, 0.7, 1.0, alpha),
    )
    for index, alpha in enumerate((0.08, 0.12, 0.18, 0.28))
]

for ghost, offset in zip(ghosts, (0.4, 0.3, 0.2, 0.1)):
    state: ke.animation.SkeletonState = motion.sample(
        time=current_time - offset
    )
    ghost.apply_state(state=state)
    ghost.set_casts_shadow(enabled=False)
```

`color=` is a per-instance multiplier; `material=` replaces the instance
material. They can be used together:

```python
ghost: ke.visual.SkinnedSurface | ke.visual.ArticulatedSurface = (
    surface.create_instance(
        path="/character/ghost",
        material=ghost_material,
        color=(0.3, 0.7, 1.0, 0.2),
    )
)
ghost.remove()
```

Removing an instance does not remove its shared assets or other instances.

## Skeleton-only visualization

Use `SkeletalVisual` to display joints and bones without a character mesh:

```python
config: ke.visual.SkeletalVisualConfig = ke.visual.SkeletalVisualConfig(
    show_joints=True
)
skeleton: ke.visual.SkeletalVisual = ke.visual.SkeletalVisual.define(
    app=self,
    material=material,
    path="/skeleton",
    state=state,
    config=config,
)
skeleton.apply_state(state=next_state)
```

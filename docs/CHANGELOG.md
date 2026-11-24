# Changelog

## Unreleased

### Added

- Support fixed-topology CPU mesh deformation through `update_geometry()` on ovrtx.
- Support CPU/CUDA ExternalBuffer instance transforms and independent skinned
  poses on ovrtx while sharing source meshes.
- Support viewport picking and selection outlines on ovrtx.
- Support Color/Phong materials, unlit, alpha masking, normal maps, and textures
  from pixel arrays on ovrtx.
- Add base-color and normal textures to `ColorMaterial`.
- Add `Material.clone()` to copy parameters while sharing textures.
- Add `asset.load_ply()` for ASCII and binary PLY meshes and colored point clouds.
- Add optional normals and linear RGB vertex colors to `MeshData.from_arrays()`,
  with vertex colors supported on static and skinned meshes.
- Add `unlit` rendering to ColorMaterial, PhongMaterial, and PBRMaterial.

### Changed

- Apply imported FBX textures to material copies, preserving the supplied material.
- Rename `set_texture()` to `set_batch_texture()` to clarify its batch-wide scope;
  keep the previous name as an alias.
- Change object selection from Shift + left click to left click. Clicking empty
  space clears selection; Shift + drag applies force in Force mode.
- Change viewport picking from AABB intersections to mesh surface picking at
  the current animated pose.
- Rename `app.ray_pick()` to `app.raycast()`, retaining the old name as an alias.
  World-space rays now intersect triangle surfaces for undeformed meshes with
  CPU transforms; skinned/deformed meshes and GPU-only transforms are excluded.
- Change ESC-triggered exit from immediate shutdown to a confirmation dialog.
  Choose Yes or press Enter to exit.
- Rename material APIs from `VertexColorMaterial`, `VertexColorStyle`, and
  `app.create_vertex_color_material()` to `ColorMaterial`, `ColorStyle`, and
  `app.create_color_material()`. Previous names remain available as aliases.

### Fixed

- Release skinned GPU buffers when their surfaces are removed.
- Improve registration time for large numbers of static objects.
- Disable gizmo editing for externally positioned debug geometry.
- Register source meshes consistently when creating renderable scene objects.
- Keep SceneGraph render options independent between objects sharing a mesh
  and material.
- Keep cloned skinned surfaces' poses and vertex updates independent.
- Preserve unset colors when cloning skinned surfaces and retain directly
  supplied materials when registering skinned meshes.
- Preserve vertex colors when merging articulation visual meshes.
- Apply instance alpha to alpha-masked shadows.
- Fix Python callbacks in `Prim.traverse()`.

## 0.3.0

### Added

- Generic CPU/CUDA `BatchFSM` with per-environment states and prioritized transitions.
  See [Batch State Machine](utils/BATCH_FSM.md).
- CPU/CUDA batch motion matching with per-environment playback, transitions,
  filtered reference-track placement, and future targets for PhysX, Newton, and MuJoCo.
  See [Motion Matching](animation/MOTION_MATCHING.md#batch-targets-for-imitation-learning).
- Batch playback hold/resume with zero velocities while held, plus Newton G1
  examples for authored start/stop clips and holding only after stopping.
- Typed motion-state queries, single-clip frame views, optional `torch.compile()`,
  and batched FK from supported PhysX joint states.
  See [Articulation Motion](animation/ARTICULATION_MOTION.md#reference-motion-library).
- Shared `app.debug` access to overlays, geometry, and text, with overlay
  visibility, removal, arrows, and depth controls. Previous entry points remain
  available. See [Debug Visualization](scene_rendering/DEBUG_VISUALIZATION.md).
- Convex decomposition caching and mesh-to-MJCF export tools.
  See [Convex Decomposition](tools/CONVEX_DECOMPOSITION.md).
- Contact snapshots and expanded collision comparison visualization.
  See [Contact Sensors](simulation/CONTACT_SENSORS.md).

### Changed

- Motion adapters prepare engine coordinates once; sampling and FK use batched
  tensor operations. Body kinematics are now precomputed by default.
- With an adapter, body targets come from motion projected onto the robot's
  joints. Source skeleton tracks remain available separately. Interpolated body
  targets need not equal FK of interpolated joint values.
- Native scalar joint coordinates and velocities interpolate linearly;
  quaternion fields use SLERP. Scalar angles are not wrapped to the shortest path.
  See [Interpolation](animation/ARTICULATION_MOTION.md#interpolation).

### Upgrade notes

Update motion-library calls from the **0.2.0 API**:

| 0.2.0 | 0.3.0 |
| --- | --- |
| `sample()` / `MotionSample` | `sample_skel_state()` / `SkeletonMotionState`; use `sample_arti_state()` for the former `backend_state` |
| `sample_kinematics()` / `MotionKinematics` | `sample_body_state()` / `BodyMotionState` |
| Separate joint and body queries | `sample_robot_state()` returns `(arti, body)` with a shared frame lookup |
| `sample_frames(..., frame_indices=...)` | Pass `frame_ids=...` to the appropriate state query |
| `packed_buffers()` | `source_buffers()` for source data; `adapted_buffers()` for typed engine data; `body_buffers()` for body targets |
| `motion(id)` returning a source clip | `source_motion(id)` with `keep_source_motions=True`; `motion(id)[frame_idx]` now samples a clip view |
| `PhysXMotionTensorSample`, `NewtonMotionTensorSample`, `MuJoCoMotionTensorSample` | `PhysXMotionState`, `NewtonMotionState`, `MuJoCoMotionState` |

Motion fields use shorter names, such as `root_positions` -> `root_pos`,
`local_rotations_wxyz` -> `local_quat_wxyz`, and PhysX `joint_positions` ->
`joint_pos`. Source `global_positions` and sampled `body_positions` become
`body_pos_w`; the other body fields are `body_quat_wxyz_w`, `body_vel_w`, and
`body_ang_vel_w`. The final `_w` denotes world coordinates. Use `body_buffers()`
for robot targets and `source_buffers()` for original skeleton tracks.

`MotionMatchingIndex` is renamed to `MotionMatchingSearch`. C++ callers should
include `animation/motion_matching_search.hpp`.

## 0.2.0

### Added

- Motion matching databases and playback controllers, pose-history queries,
  and kinematic foot-contact detection.
  See [Motion Matching](animation/MOTION_MATCHING.md) and [Foot Contact](animation/FOOT_CONTACT.md).
- Rigid assets with separate visual and collision geometry, material support,
  and batched terrain/rigid ray queries.
  See [Rigid Bodies](simulation/RIGID_BODIES.md).
- Runtime mass, center-of-mass, and collision-material updates for domain
  randomization, plus body/shape mapping APIs.
  See [Mass Properties](simulation/MASS_PROPERTIES.md) and [Collision Shapes](simulation/COLLISION_SHAPES.md).
- D6 joints, procedural articulations, and articulation joint wrench/effort
  readback. See [D6 Joints](simulation/D6_JOINTS.md) and [Articulation](simulation/ARTICULATION.md).
- USD material and texture loading, scene lighting and camera import, and
  opacity textures. See [Load Assets](scene_rendering/LOAD_ASSETS.md).
- Local MCP inspection, scene editing, frame capture, recording, and profiling.
- Motion conversion/viewing tools and isolated wheel validation for Linux and macOS.

### Changed

- Improved MJCF/URDF import, fixed articulation frames, and visualization of
  attachments and links without meshes.
- Updated macOS ARM64 PhysX support to 5.8 and corrected lighting/selection guides.

(upgrade-notes-0-2-0)=
### Upgrade notes

**Behavior changes (no error, different result)**

- Directional light `direction` now follows emitted rays. Negate existing
  source-facing vectors, including those passed to `App.set_light_direction()`.
  Spot direction is unchanged.
- Directional, Point, and Spot light getters now return local values. Remove
  world-to-local conversion applied to their results.
  See [Lighting](scene_rendering/LIGHTING.md).
- Compound rigid `body_*` arrays now contain one body per actor, rather than
  one slot per collider. Use `get_collision_shape_pose()` for collider poses;
  even a single offset collider differs from its body. Mass, force, and contact
  state refer to the actor. See [Collision Shapes](simulation/COLLISION_SHAPES.md).

**Removed or renamed APIs**

- `render_frame_once()` is removed. Use `update_and_draw()`, or
  `process_events()` + `draw_frame()` when an external loop steps simulation.
- Quaternion API names now specify their order, e.g. `set_local_rotation()` ->
  `set_local_quat_wxyz()`. Skeleton quaternion setters use WXYZ; update values
  previously supplied in XYZW order.
- MJCF import rejects unsupported joint types, including `slide`.
  Supported types are `hinge`, `ball`, and `free`.

## 0.1.0

Initial release.

# Examples

## Custom rendering

- `python/examples/custom_graphics_pipeline.py` draws an interactive gradient
  triangle through a custom graphics pipeline and exposes its shader and
  pipeline definitions in the Resource panel.

## Recommended first runs

| Goal | Command from the repository root | Requirements | Expected result |
|---|---|---|---|
| First scene | `python ./python/examples/render_prim_scene.py` | CPU renderer | Ground, box, sphere |
| First simulation | `python ./python/examples/sim_world_minimal.py` | PhysX CPU | Falling ball with reset UI |
| Batched simulation | `python ./python/examples/sim_world_multi_env.py --num-envs 16` | PhysX CPU, Torch | Multiple ramp environments |
| Procedural terrain | `python ./python/examples/view_procedural_terrain.py` | PhysX CPU | Tiled terrain with falling collision bodies |
| [Interactive stair raycasts](../simulation/PROCEDURAL_TERRAIN.md#try-the-interactive-stair-raycast-example) | `python python/examples/asset/rigid_asset.py` | PhysX CPU, Torch, Warp | Procedural stairs with independent poses and raycast visualization |
| MJCF control | `python ./python/examples/mjcf_dof_control.py /path/to/robot.xml` | MJCF asset | DOF sliders and robot |
| BVH motion | `python ./python/examples/view_bvh_character.py /path/to/motion.bvh` | BVH asset | Skeleton and sequencer |
| Motion Matching | `python python/examples/animation/motion_matching_lafan.py --dataset "$LAFAN_DIR"` | LAFAN-resolved BVHs + Geno bind FBX | Skinned Geno walking, running, stopping and turning |
| G1 Motion Matching | `python python/examples/animation/motion_matching_g1.py --dataset "$SEED_DIR/g1/csv"` | BONES-SEED G1 CSVs + SciPy | G1 rigid-link playback with joint limits; WASD/gamepad input when autoplay is off |
| GPU contacts | `python ./python/examples/sim_gpu_contact_sensor.py --viewer` | Linux/NVIDIA | CUDA contact visualization |

See [Motion Matching](../animation/MOTION_MATCHING.md) for dataset setup,
shared controls, caching, filtering, custom features, and search benchmarks.

`python/examples/keyboard_visualizer.py` previews the keyboard texture and key
regions used by the MM input overlay.

## Asset viewers

- `view_obj_scene.py`: OBJ/MTL material subsets and textures.
- `view_fbx_mesh.py`: static FBX meshes and textures.
- `view_fbx_character.py`: skinned FBX character.
- `view_motion_party.py`: BVH, FBX, and SMPL/AMASS motions playing
  together through their standard visual APIs and sequencers.
- `view_smpl_motion.py`: SMPL surface motion with a sequencer.
- `view_usd_scene.py`: optional USD-enabled development build.
- `usd_file_bridge.py`: Python OpenUSD file exchange.

## Debug and validation examples

Files under `python/examples/smoke/` validate API and runtime behavior. They are
maintainer checks, not the recommended way to learn the package.

Advanced native examples live under `examples/`; legacy examples are retained
for regression and historical reference.

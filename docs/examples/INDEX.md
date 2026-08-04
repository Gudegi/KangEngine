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
| MJCF control | `python ./python/examples/mjcf_dof_control.py /path/to/robot.xml` | MJCF asset | DOF sliders and robot |
| BVH motion | `python ./python/examples/view_bvh_character.py /path/to/motion.bvh` | BVH asset | Skeleton and sequencer |
| GPU contacts | `python ./python/examples/sim_gpu_contact_sensor.py --viewer` | Linux/NVIDIA | CUDA contact visualization |

## Asset viewers

- `view_obj_scene.py`: OBJ/MTL material subsets and textures.
- `view_fbx_mesh.py`: static FBX meshes and textures.
- `view_fbx_character.py`: skinned FBX character.
- `view_usd_scene.py`: optional USD-enabled development build.
- `usd_file_bridge.py`: Python OpenUSD file exchange.

## Debug and validation examples

Files under `python/examples/smoke/` validate API and runtime behavior. They are
maintainer checks, not the recommended way to learn the package.

Advanced native examples live under `examples/`; legacy examples are retained
for regression and historical reference.

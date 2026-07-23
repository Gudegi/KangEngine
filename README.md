# KangEngine

A lightweight C++/Python engine for visualizing motion, robotics assets, and PhysX-based simulation experiments.

> **Note:** This is a personal, long-term codebase dedicated to ongoing research and self-study. It is crafted as a lifetime sandbox for exploring computer graphics (especially character animation) and robotics control.

![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-lightgrey)
![Linux](https://img.shields.io/badge/Linux-Ubuntu%2024.04-orange)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PhysX](https://img.shields.io/badge/Physics-PhysX-green)

<table align="center">
  <tr>
    <td width="33%" align="center"><img src="images/instanced_robots.png" alt="Instanced robot simulation" style="width:100%;"></td>
    <td width="33%" align="center"><img src="images/joint_monkey.png" alt="Robot joint control example" style="width:100%;"></td>
    <td width="33%" align="center"><img src="images/ball_collisions.png" alt="Rigid body collision example" style="width:100%;"></td>
  </tr>
</table>

KangEngine is built for quick iteration around character motion, robot assets, and simulation visualization. The C++ side owns the renderer, scene graph, asset loaders, and PhysX integration; the Python package exposes the same runtime for scripts, motion tools, control experiments, and MimicKit integration.

## Requirements

- **OS:** macOS on Apple Silicon (Tahoe tested) or Linux (Ubuntu 24.04 tested)
- **Build:** C++17, CMake, vcpkg, PhysX 5.1/5.8
- **Python:** 3.12 for bindings and examples
- **Graphics & GPU:** OpenGL 4.1+ compatible GPU (NVIDIA GPU required only for PhysX GPU/CUDA workflows)

## What It Does

- **Multi-Format Asset Viewer:** Loads FBX, BVH, MJCF, OpenUSD, OBJ, and STL assets.
- **Motion Inspection:** Visualizes skeletal motion, FK poses, root trajectories, contacts, and tracking targets.
- **Interactive Visualization:** Provides skinned character rendering, skeleton overlays, debug drawing, and scene interaction tools.
- **PhysX Simulation:** Runs rigid bodies, articulated robots, and motion-tracking control experiments.
- **Python Workflows:** Provides Python APIs for motion editing, IK/control experiments, simulation scripts, and MimicKit integration.

## Quick Start

Build the C++ executable:

```bash
cmake --preset=vcpkg
cmake --build build/release
make run2
```

Build and install the Python package:

```bash
uv venv python/.venv --python 3.12
source python/.venv/bin/activate
make build_python
uv pip install -e ./python
```

Run a Python motion viewer:

```bash
python ./python/examples/view_bvh_character.py
```

Run a Python PhysX example:

```bash
python ./python/examples/sim_world_minimal.py
```

See [Build Guide](docs/BUILD.md) for platform setup, PhysX, USD, and Python binding details.

## Feature Overview

### Rendering

- OpenGL renderer with instanced mesh drawing and a lightweight graphics abstraction layer.
- SceneGraph and ExternalBuffer transform paths for authored scenes and high-throughput simulation visuals. See `examples/physics/physx_h1_instancing.cpp`.
- Shadow mapping, skybox rendering, gamma post-processing, ImGui tooling, and post-process selection outlines.
- GPU skinning for animated FBX characters.
- Debug rendering utilities for lines, arrows, coordinate axes, points, and camera frustums.

### Asset Import

- FBX: skeletons, animation clips, static meshes, and skinned meshes.
- BVH: skeleton hierarchy, frame time, root motion, and local joint rotations.
- MJCF: articulated characters, collision geometry, joints, and inertials.
- USD (optional development build): mesh traversal, material subsets, and diffuse texture loading. Distributed wheels do not currently include USD support.
- OBJ/STL static mesh import.

### Simulation & Animation

- PhysX rigid bodies and articulated robot simulation.
- Skeleton trees, sampled motion clips, FK, pose states, and skeleton visual bridges.
- Bridges for syncing physics, skeletons, and skinned characters to scene/render state.
- Experimental XPBD cloth simulation. See `examples/physics/xpbd_cloth.cpp`.

### Advanced Runtime Paths

- **ExternalBuffer visual sync:** skips per-object scene graph mutation during simulation and uploads batched transforms directly to renderer-owned instance buffers. This is the preferred path for large simulation visuals such as many robot bodies or rigid objects.
- **XPBD cloth:** a non-PhysX cloth simulation experiment used to explore constraint-based deformable simulation.

### Python

- pybind11 bindings for app, scene, animation, physics, asset, and renderer-facing APIs.
- Headless simulation and live visualization helpers.
- Motion editor modules for trajectories, contacts, targets, and tracking overlays.
- MimicKit-compatible backend adapter.

See [Simulation API](docs/SIMULATION_API.md) for the recommended `KangSimWorld` workflow and how it relates to lower-level PhysX wrappers.

## Development Notes

This project is evolving quickly. While the main workflows are stable, some internal and high-level APIs are under active development and subject to change.

Planned work:

- WebGPU backend implementation beyond the current placeholder.

## RL With MimicKit

KangEngine can be used as a backend engine of [MimicKit](https://github.com/xbpeng/MimicKit) through KangEngine's Python package. Use the `backend_kangengine` branch of MimicKit and keep MimicKit in a separate Python environment.

**Note:** GPU contact sensors provide contact count, contact state, and
accumulated normal impulse and force. Tangential friction impulse and a full
six-axis contact wrench are not currently exposed as sensor outputs.

<p align="center">
  <img src="images/Mimickit_kangengine_1.png" alt="MimicKit running with KangEngine" style="width:70%;">
</p>

<details>
<summary>MimicKit setup and run commands</summary>

1. Clone the KangEngine-enabled MimicKit fork branch.

    ```bash
    git clone -b backend_kangengine https://github.com/Gudegi/MimicKit.git
    ```

2. Create and activate a MimicKit Python environment with uv.

    ```bash
    cd MimicKit
    uv venv .venv --python 3.12
    source .venv/bin/activate
    ```

3. Build KangEngine's Python extension from the KangEngine repo.

    ```bash
    cd /path/to/KangEngine
    make build_python
    ```

4. Install KangEngine's Python package into the MimicKit environment.

    ```bash
    uv pip install -e ./python
    ```

5. Install MimicKit dependencies.

    ```bash
    cd /path/to/MimicKit
    uv pip install -r requirements.txt
    ```

6. Run a small motion visualization test.

    ```bash
    python mimickit/run.py \
      --mode test \
      --num_envs 1 \
      --engine_config data/engines/kangengine_engine.yaml \
      --env_config data/envs/view_motion_humanoid_env.yaml \
      --visualize true \
      --devices cpu \
      --test_episodes 10
    ```

7. Run pretrained policy inference with KangEngine.

    ```bash
    python mimickit/run.py \
      --mode test \
      --num_envs 4 \
      --engine_config data/engines/kangengine_engine.yaml \
      --env_config data/envs/amp_humanoid_env.yaml \
      --agent_config data/agents/amp_humanoid_agent.yaml \
      --visualize true \
      --model_file data/models/amp_humanoid_spinkick_model.pt
    ```

8. Train an AMP policy with KangEngine.

    ```bash
    python mimickit/run.py \
      --mode train \
      --num_envs 4096 \
      --engine_config data/engines/kangengine_engine.yaml \
      --env_config data/envs/amp_humanoid_env.yaml \
      --agent_config data/agents/amp_humanoid_agent.yaml \
      --visualize false \
      --out_dir output/
    ```

For reference, the MimicKit KangEngine backend uses an engine config like this:

```yaml
engine_name: "kangengine"

control_mode: "pos"
control_freq: 30
sim_freq: 120
env_spacing: 5
enable_self_collisions: false
```

The `backend_kangengine` branch already includes `data/engines/kangengine_engine.yaml`, so you usually do not need to create it manually.

</details>

## References

KangEngine is inspired by and built upon ideas from these excellent projects:

- [MimicKit](https://github.com/xbpeng/MimicKit): Motion imitation and RL experiment structure.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): Robot learning workflows and simulation tooling.
- [Newton](https://github.com/newton-physics/newton): Robotics API shape and GPU simulation design.
- [SAPIEN](https://github.com/haosulab/SAPIEN): Robotics simulation and GPU simulation design.
- [GenoViewPython](https://github.com/orangeduck/GenoViewPython): Skeletal animation and motion-debug visualization.
- [AI4AnimationPy](https://github.com/facebookresearch/ai4animationpy): Motion modules and animation-engine structure.
- [NVIDIA PhysX](https://github.com/NVIDIA-Omniverse/PhysX): Rigid body and articulation simulation.
- [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD): Prim/path concepts and optional USD asset interchange.

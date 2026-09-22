# KangEngine

**Character animation and GPU-parallel physics simulation in C++ and Python.**<br>
Built for robot learning, motion workflows, and interactive visualization.

[![Official Documentation](https://img.shields.io/badge/Docs-Official%20Documentation-blue)](https://gudegi.github.io/KangEngine/)
[![PyPI](https://img.shields.io/pypi/v/kangengine)](https://pypi.org/project/kangengine/)
![Python](https://img.shields.io/badge/Python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

![Parallel robotics simulation in KELab](docs/images/KELAB/demo2.png)

| Task | Framework | Environments | Environment Step FPS | Step + Inference FPS | Step + Inference + Train FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Cartpole | KELab (KangEngine) | 4096 | 1,153,317 | 914,046 | 527,070 |
| Cartpole | IsaacLab v2.3.2 (local) | 4096 | 978,112 | 794,296 | 491,239 |
| G1 Rough | KELab (KangEngine) | 4096 | 104,009 | 95,889 | 84,954 |
| G1 Rough | IsaacLab v2.3.2 (local) | 4096 | 85,039 | 85,122 | 76,552 |

GPU-parallel simulation and training performance benchmarks.<br>
Headless, RTX 4090 + i7-13700K; FPS counts environment control steps.
Single runs with differing software stacks and simulation implementations.
[Benchmark details](https://gudegi.github.io/KangEngine/guide/KELAB_BENCHMARKS.html).

## Features

- **GPU-parallel simulation:** PhysX rigid bodies and articulated robots across
  thousands of environments on Linux/NVIDIA.
- **Robot learning:** Train policies with [KELab](https://gudegi.github.io/KangEngine/guide/KELAB.html),
  a reinforcement-learning framework built on KangEngine with RL-Games and RSL-RL
  support, or use KangEngine as a simulation backend for
  [MimicKit](https://gudegi.github.io/KangEngine/guide/MIMICKIT.html).
- **Character motion:** BVH/FBX motion visualization, SMPL/AMASS workflows,
  skeletal animation, and retargeting.
- **Visualization:** Interactive C++/OpenGL rendering for KangEngine and
  [Newton](https://gudegi.github.io/KangEngine/guide/NEWTON.html) simulation workflows.

## Quick Start

Use Python 3.12 in a virtual environment:

```bash
uv venv --python python3.12
source .venv/bin/activate
uv pip install kangengine
```

- **OS:** macOS 26+ (Apple Silicon), Ubuntu 24.04 (x86-64)
- **Python:** 3.12
- **Simulation:** PhysX 5.1 CPU on macOS; PhysX 5.8 CPU/GPU on Ubuntu
- **Dependencies:** OpenGL 4.1+ for visualization; NVIDIA driver 580+ for CUDA 13 on Ubuntu

See [Installation](https://gudegi.github.io/KangEngine/guide/getting_started/INSTALLATION.html) for details.

```python
import kangengine as ke


class HelloApp(ke.App):
    def setup(self):
        self.materials = self.create_standard_materials()
        self.scene.add_ground("/ground", scale=10.0)
        box = self.scene.add_mesh(
            "/box",
            ke.geometry.create_cube_data(1.0),
            self.materials.pbr,
            color=ke.Vec4(0.8, 0.3, 0.02, 1.0),
        )
        box.set_local_translation(ke.Vec3(0.0, 0.5, 0.0))
        self.set_camera_view([3.0, 2.5, 4.0], [0.0, 0.5, 0.0])


app = HelloApp()
app.initialize(width=1920, height=1080, hide_ui=False, up_axis=ke.UpAxis.Y)
app.start()
```

See [Hello App](https://gudegi.github.io/KangEngine/guide/getting_started/HELLO_APP.html) for lifecycle callbacks.

## Examples

| Workflow | Start here |
| --- | --- |
| Create a scene and manipulate objects | [Hello App](https://gudegi.github.io/KangEngine/guide/getting_started/HELLO_APP.html) / [First Scene](https://gudegi.github.io/KangEngine/guide/getting_started/FIRST_SCENE.html) |
| Drop a ball with physics | [First Simulation](https://gudegi.github.io/KangEngine/guide/getting_started/FIRST_SIMULATION.html) |
| Load and simulate articulated robots | [Articulation](https://gudegi.github.io/KangEngine/guide/simulation/ARTICULATION.html) |
| Load and play character motion | [Load Motion](https://gudegi.github.io/KangEngine/guide/animation/LOAD_MOTION.html) |
| Train RL policies with KELab | [KELab](https://gudegi.github.io/KangEngine/guide/KELAB.html) / [Benchmarks](https://gudegi.github.io/KangEngine/guide/KELAB_BENCHMARKS.html) |
| Train RL policies with Mimickit  | [MimicKit](https://gudegi.github.io/KangEngine/guide/MIMICKIT.html) |
| Visualize Newton simulations | [Newton integration](https://gudegi.github.io/KangEngine/guide/NEWTON.html) |

Complete scripts are in [python/examples](python/examples). Clone the repository
to run those files; they are separate from the installed Python package.

## License

KangEngine is licensed under the [MIT License](LICENSE).
Bundled third-party components retain their own licenses.

## Acknowledgments

Built with [NVIDIA PhysX](https://github.com/NVIDIA-Omniverse/PhysX) and
[OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD), with inspiration from
[MimicKit](https://github.com/xbpeng/MimicKit),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab),
[Newton](https://github.com/newton-physics/newton),
[SAPIEN](https://github.com/haosulab/SAPIEN),
[GenoViewPython](https://github.com/orangeduck/GenoViewPython), and
[AI4AnimationPy](https://github.com/facebookresearch/ai4animationpy).

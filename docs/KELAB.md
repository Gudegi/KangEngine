# KELab

| | |
|:---:|:---:|
| ![KELab parallel humanoid environments](../../images/KELAB/demo1.png) | ![KELab humanoid simulation debug view](../../images/KELAB/demo2.png) |

[KELab](https://github.com/Gudegi/KELab) is a reinforcement-learning framework
built on KangEngine. It connects KangEngine simulation and rendering to
Gymnasium tasks and training workflows for RL Games, RSL-RL, and MimicKit.

KELab is maintained as a separate project. Use its repository for task and
training documentation; this page covers the shortest KangEngine setup path.

## Install

Until KangEngine wheels are available, build KangEngine and install both
projects into the KELab environment:

```bash
# KangEngine
cd /path/to/KangEngine
make build_python_cuda

# KELab
cd /path/to/KELab
uv venv --python python3.12
source .venv/bin/activate
uv pip install -e /path/to/KangEngine/python
uv pip install -e ".[rl-games]"
```

Verify the installation:

```bash
python -c "import kangengine, ke_lab, ke_lab_tasks, ke_lab_rl"
```

## Run an environment

Run a registered task with random actions and KangEngine rendering:

```bash
python scripts/random_agent.py \
  --task Humanoid-v0 \
  --num_envs 4 \
  --device cuda \
  --render
```

For headless training, increase `--num_envs`, omit `--render`, and use one of
the training scripts under `scripts/rl_games`, `scripts/rsl_rl`, or
`scripts/mimickit`.

KELab currently targets PhysX, with its high-throughput training path intended
for Linux systems with NVIDIA CUDA. CPU execution remains useful for small
smoke tests and debugging.

See the [KELab repository](https://github.com/Gudegi/KELab) for available
tasks, training commands, and optional integrations.

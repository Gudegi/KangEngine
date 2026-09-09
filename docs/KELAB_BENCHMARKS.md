# Performance Benchmarks

KELab benchmarks run headless on an NVIDIA GeForce RTX 4090 and Intel Core
i7-13700K with 4096 environments. FPS counts environment control steps.

KangEngine revision: [`cef3dd2`](https://github.com/Gudegi/KangEngine/commit/cef3dd2981c9bbc5dc81d1045efc079e326866d9).

## Simulation and Training Throughput

| Task | Framework | Environments | Environment Step FPS | Step + Inference FPS | Step + Inference + Train FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Cartpole | KELab | 4096 | 1,153,317 | 914,046 | 527,070 |
| Cartpole | IsaacLab v2.3.2 (local) | 4096 | 978,112 | 794,296 | 491,239 |
| G1 Rough | KELab | 4096 | 104,009 | 95,889 | 84,954 |
| G1 Rough | IsaacLab v2.3.2 (local) | 4096 | 85,039 | 85,122 | 76,552 |

- **Cartpole:** RL-Games, 50 training iterations.
- **G1 Rough:** RSL-RL 5.4.2, 50 training iterations; environment-only FPS is
  measured separately over 100 steps with random actions.
- Following IsaacLab v2.3.2 benchmark scripts, values are arithmetic means of
  native per-iteration FPS (per-step FPS for non-RL), with no warm-up exclusion
  or added CUDA synchronization. Inference includes rollout collection.
- Local IsaacLab runs use the fresh pinned reference environment on the same
  GPU/CPU, with matching seed 42 and run lengths. These are single runs;
  Torch/Python versions and simulation implementations differ between frameworks.

For reference, [IsaacLab v2.3.2 publishes](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/reinforcement-learning/performance_benchmarks.html)
the following RTX 4090 results with a Ryzen 9 7950X CPU. CPU and measurement
settings differ from the local runs. Its published table does not specify the
training iteration count; the local measurement length is given above.

| IsaacLab Environment | Environments | Environment Step FPS | Step + Inference FPS | Step + Inference + Train FPS |
| --- | ---: | ---: | ---: | ---: |
| Isaac-Cartpole-Direct-v0 | 4096 | 1,100,000 | 910,000 | 510,000 |
| Isaac-Velocity-Rough-G1-v0 | 4096 | 94,000 | 88,000 | 82,000 |

## Humanoid Training Time

RL-Games, 4096 environments × 32 rollout steps × 500 iterations:
**65,536,000 environment steps** on RTX 4090.

| RL Library | KELab — Torch (s) | KELab — Warp base (s) | IsaacLab v2.3.2 Published Time (s) |
| --- | ---: | ---: | ---: |
| RL-Games | **193.80** | **176.65** | **201** |

Both KELab compute backends were measured on 2026-09-09 with Torch compilation
disabled. Times are script-reported
training durations. The [IsaacLab reference](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/reinforcement-learning/rl_frameworks.html#training-performance)
matches GPU model and step count; other system and task settings may differ.

## G1 Rough: 4000-Iteration Training

Both frameworks ran locally with RSL-RL 5.4.2, matching PPO settings, seed 42,
and 4096 environments × 24 rollout steps × 4000 iterations:
**393,216,000 environment steps**. KELab uses the reference USD robot with
37 DOFs, three convex hulls and implicit PD.

| Metric | KELab | IsaacLab v2.3.2 |
| --- | ---: | ---: |
| Rollout collection time (s) | 3,514.75 | 3,927.76 |
| PPO update time (s) | 452.50 | 445.07 |
| Total measured training time (s) | **3,967.25** | **4,372.83** |
| Full-run training FPS | **99,116** | **89,923** |
| Final mean episode reward | 29.72 | 26.75 |
| Final mean episode length (s) | 19.69 | 19.71 |

Training FPS is total steps divided by summed collection and update times,
excluding initialization and checkpoint saving. Reward and episode length
are averages over the final 50 iterations. Results are from one seed;
Python, Torch and physics backends differ between frameworks.

## Benchmark Commands

The KELab repository also provides `scripts/benchmarks/isaaclab_reference/`
with pinned dependencies and standalone IsaacLab v2.3.2 train/play commands.
Its README describes a fresh conda setup using Isaac Sim 5.1.0 and RSL-RL 5.4.2.

Run from the KELab repository:

```bash
# Cartpole throughput during RL-Games training
python scripts/benchmarks/benchmark_rl_games.py \
  --task Cartpole-Direct-v0 --num_envs 4096 --max_iterations 50 \
  --output logs/benchmarks/cartpole.json

# G1 rollout and training throughput
python scripts/benchmarks/benchmark_rsl_rl.py \
  --task G1-Velocity-Rough-v0 --num_envs 4096 --max_iterations 50 \
  --output logs/benchmarks/g1_training.json

# G1 environment-only throughput
python scripts/benchmarks/benchmark_env.py \
  --task G1-Velocity-Rough-v0 --num_envs 4096 --num_frames 100 \
  --output logs/benchmarks/g1_env.json

# Humanoid training time
python scripts/rl_games/train.py --task Humanoid-v0 \
  --num_envs 4096 --device cuda:0 --compute_backend torch \
  --max_iterations 500

# Humanoid training time — Warp base
python scripts/rl_games/train.py --task Humanoid-v0 \
  --num_envs 4096 --device cuda:0 --compute_backend warp \
  --max_iterations 500

# G1 Rough training
python scripts/rsl_rl/train.py --task G1-Velocity-Rough-v0 \
  --num_envs 4096 --device cuda:0 --seed 42 --max_iterations 4000
```

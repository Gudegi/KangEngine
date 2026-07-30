# Simulation Run Modes

`SimulationTimingConfig` defines rates and time intervals. `SimulationRunConfig`
defines how an externally stepped loop relates those intervals to wall time.
The supported run modes are:

| Mode | Rendering | Wall-clock wait | Catch-up | Primary use |
|---|---|---|---|---|
| `HEADLESS_FAST` | Disabled | No | No | RL training and benchmarks |
| `OFFSCREEN_FAST` | Offscreen, on request | No | No | Image capture, evaluation, and datasets |
| `PACED` | Human window | Yes | No | Wall-clock-paced policy playback (real-time pace) |

**Wall-clock wait** means that a step finishing faster than its configured
`step_dt` waits for the remaining real-world time. **Catch-up** means executing
multiple steps after a delay to recover the difference between simulation time
and wall-clock time. PACED waits when it is ahead but does not catch up when it
falls behind.

Create an explicit policy with:

```python
run_config = ke.SimulationRunConfig(mode="paced")
```

## KELab defaults

`DirectRLEnv` chooses a mode from Gym's `render_mode` when no explicit
`run_config` is supplied:

```text
render_mode=None          -> HEADLESS_FAST
render_mode="human"       -> PACED
render_mode="rgb_array"   -> OFFSCREEN_FAST
```

Each mode has one rendering configuration. `HEADLESS_FAST` requires
`render_mode=None`, `OFFSCREEN_FAST` requires `render_mode="rgb_array"`, and
`PACED` requires `render_mode="human"`. Human rendering is performed
automatically after each reset and environment step. RGB array rendering
occurs without wall-clock waiting when the caller invokes `env.render()`.

PACED renders per environment step, not per physics substep:

```text
environment step
├─ apply action
├─ physics step x decimation
├─ observations, rewards, termination, and reset
├─ render latest state
└─ wait until step_dt has elapsed
```

The environment rates are:

```text
physics_hz = 1 / sim.dt
step_dt = sim.dt * decimation
step_hz = control_hz = 1 / step_dt
```

Current task examples:

| Task | Physics rate | Decimation | PACED target |
|---|---:|---:|---:|
| `Humanoid-v0` | 120 Hz | 2 | 60 Hz |
| `Pi-Locomotion-v0` | 1000 Hz | 20 | 50 Hz |

`SimulationPacer` waits only when processing finishes ahead of the next
deadline. If policy inference, physics, or rendering exceeds `step_dt`, the
loop does not issue multiple catch-up steps; playback becomes slower than wall
time. This matches the MimicKit-style playback behavior and avoids catch-up
bursts in an externally stepped loop.

## Native App scheduling

`SimulationRunConfig` does not select the native `App` scheduler. `App.start()`
uses `FixedStepClock`, which accumulates wall time and may invoke
`fixedUpdate()` zero or more times per rendered frame. This is appropriate for
interactive simulations whose physics should remain independent of rendering.

Do not use `SimulationPacer` inside an App `fixedUpdate()` callback. Keep App
physics in `fixedUpdate()` and let `FixedStepClock` schedule it.

`renderFrameOnce()` is supported for manually driven rendering. Despite its
name, it currently processes input and advances App frame scheduling in
addition to drawing. Do not treat it as a render-only function or call it from
inside `fixedUpdate()`.

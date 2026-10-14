# Simulation Run Modes

Choose the execution path based on who advances the simulation:

- **KangEngine App:** `app.start()` or `app.update_and_draw()` uses
  `FixedStepClock` to schedule `fixed_update()` callbacks from elapsed wall time.
- **KELab:** `env.step()` advances the simulation explicitly.
  `SimulationRunConfig` selects rendering and wall-clock waiting behavior.

An App can also display an externally stepped simulation using
`process_events()` and `draw_frame()`. See
[Fixed Timestep and Rendering](FIXED_TIMESTEP.md)
for the distinction between `SimulationTimingConfig` rates and
`SimulationRunConfig` execution policy.

## Shared execution policies

`SimulationRunConfig` describes rendering and wall-clock waiting for external
loops, including KELab and manually stepped App loops. The loop or runner must
apply the policy; constructing the config does not execute steps or create a
viewer.

`mode` accepts these strings or the corresponding `ke.SimulationRunMode` values:

| Enum member | String value | Intended rendering | Wall-clock wait | Primary use |
| --- | --- | --- | --- | --- |
| `HEADLESS_FAST` | `"headless_fast"` | Disabled | No | RL training and benchmarks |
| `OFFSCREEN_FAST` | `"offscreen_fast"` | Offscreen, on request | No | Image capture, evaluation, and datasets |
| `PACED` | `"paced"` | Human window | Yes | Policy playback at real-time pace |

KELab creates its viewer according to this policy. In a manual App loop, you
choose rendering through App initialization (`headless=True` for offscreen)
and frame calls. Changing the policy neither creates nor removes a renderer,
and `HEADLESS_FAST` does not block an explicit `draw_frame()` call.
`SimulationPacer` controls waiting; `app.configure_run()` controls App recording
behavior as described below.

`SimulationPacer` implements the waiting part: in `PACED` mode, it waits when
the loop is ahead of wall time. If a step takes too long, it continues without
adding catch-up steps, so playback becomes slower than real time. In the other
modes, it does not wait. **Catch-up** means running multiple fixed updates in
one rendered frame to recover simulation time lost during a delay. The App's
`FixedStepClock` supports this, bounded by `max_catch_up_steps` and
`max_frame_delta`; see [Fixed Timestep and Rendering](FIXED_TIMESTEP.md).

## KangEngine App

### App-scheduled updates

`App.start()` repeatedly calls `update_and_draw()`. `FixedStepClock` accumulates
wall time and may invoke `fixed_update()` zero or more times per rendered frame.
Put App-owned physics updates in `fixed_update()`; `SimulationRunConfig` does
not select or replace this scheduler.

Do not use `SimulationPacer` inside an App `fixed_update()` callback. Let
`FixedStepClock` schedule the updates.

### Manually driven frames

Choose the frame method based on who owns simulation updates:

| Method | Behavior |
| --- | --- |
| `update_and_draw()` | Process events, call `pre_update()`, run due `fixed_update()` calls, then draw and present. `start()` repeats this operation. |
| `process_events()` | Poll window events and handle camera input and simulation hotkeys. No update or render callbacks. |
| `draw_frame()` | Call `pre_render()`, `render()`, and `post_render()`, render the scene/UI, capture screenshots/video, and present. No event polling or fixed-step scheduling. |

When your loop or an external simulator steps physics explicitly, sync its
visual state, then call `process_events()` and `draw_frame()`. This path does not
run the App fixed-update scheduler. For a responsive paused viewer, continue
calling both methods while simulation is paused. If simulation hotkeys are
enabled, `consume_simulation_step()` consumes a paused single-step request
exactly once; it does not step physics or require a fixed-update rate.

Manual loops call `setup()` themselves if their App subclass needs it. Render
callbacks may change state if user code explicitly does so. Do not call frame
execution methods recursively from these callbacks or from `fixed_update()`.

Input timing is independent of drawing; draw-only calls do not reset the App
update clock. Use one scheduling owner consistently to avoid wall-time catch-up
when returning to App-driven updates.

### Paced playback without KELab

Assume `app` is an initialized windowed `ke.App`, `world` is a
`ke.sim.KangSimWorld` with `sim_dt=1.0 / 120.0`, and `visual` is its populated
`ke.visual.sim.SimWorldVisualizer`. This loop advances two physics steps per
frame and targets 60 frames per second:

```python
run_config: ke.SimulationRunConfig = ke.SimulationRunConfig(mode="paced")
pacer: ke.SimulationPacer = ke.SimulationPacer(run_config=run_config)
step_dt = 2 * world.sim_dt

while not app.should_close():
    app.process_events()
    if app.should_close():
        break
    world.step(substeps=2)
    visual.sync()
    app.draw_frame()
    pacer.wait(step_dt=step_dt)
```

Call `pacer.wait()` after stepping and drawing so the steady-state wait accounts
for both costs. This loop owns simulation updates and does not call
`update_and_draw()` or `fixed_update()`.

### App recording policy

`app.configure_run()` currently selects how App video recording handles frames:

| Mode | Recording behavior |
| --- | --- |
| `PACED` | Select or repeat frames to match elapsed wall time |
| `OFFSCREEN_FAST` | Record one video frame per rendered frame |
| `HEADLESS_FAST` | Raise `RuntimeError` when recording is started |

It does not change renderer initialization, schedule simulation updates, or
wait between frames. Without an explicit policy, App uses `PACED` for a visible
window and `OFFSCREEN_FAST` for `headless=True`. The windowed example above
therefore needs no `configure_run()` call. Use
`app.configure_run(config=run_config)` to override the recording policy. See
[Video Recording](../scene_rendering/VIDEO_RECORDING.md) for FPS and capture
settings.

## KELab

KELab owns simulation stepping through `env.step()` and applies the selected
run policy to its viewer. Its viewer calls `SimulationPacer` for paced playback;
do not add another pacer around `env.step()`.

### Defaults and environment stepping

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

### Timing settings and derived values

App-driven loops configure `SimulationTimingConfig.physics_hz` and
`fixed_update_hz` directly; `physics_dt`, `fixed_dt`, and their ratio
`decimation` are derived properties. See
[Fixed Timestep and Rendering](FIXED_TIMESTEP.md) for configuration examples.

KELab starts from environment configuration fields instead:

| Value | Defined by | Meaning |
| --- | --- | --- |
| `sim.dt` | `DirectRLEnvCfg.sim.dt` | Seconds per physics step |
| `decimation` | `DirectRLEnvCfg.decimation` | Integer physics steps per environment step |
| `physics_hz` | Derived as `1 / sim.dt` | Physics rate in the environment's timing config |
| `step_dt` | Derived as `sim.dt * decimation` | Seconds per environment step; exposed as `env.step_dt` |
| `step_hz`, `control_hz` | Derived as `1 / step_dt` | Environment/control rate; exposed on `env` |

KELab builds its `SimulationTimingConfig` from these fields, with
`fixed_update_hz = 1 / step_dt`. For example, `sim.dt = 1 / 120` and
`decimation = 2` give a 60 Hz environment step and paced playback target.

## Migration from `render_frame_once()`

`render_frame_once()` has been removed. Replace it with `update_and_draw()` to
retain App scheduling, or with `process_events()` and `draw_frame()` for external
stepping. Event polling now precedes input processing in `update_and_draw()`.
User callback names are unchanged.

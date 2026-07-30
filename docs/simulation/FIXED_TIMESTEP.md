# Fixed Timestep and Rendering

KangEngine separates wall-clock scheduling, fixed simulation updates, and
render preparation. A simulation `App` should use this lifecycle:

```text
rendered frame
├─ preUpdate()             input and per-frame state changes
├─ fixedUpdate(fixed_dt)   zero or more control/physics updates
├─ preRender()             copy the latest state to visuals
├─ render()                ImGui and other drawing
└─ postRender()            frame completion
```

Configure the render, physics, and fixed-update rates together in `setup()`:

```python
def setup(self):
    self.timing = self.configure_timing(
        ke.SimulationTimingConfig(
            render_hz=60.0,
            physics_hz=120.0,
            fixed_update_hz=60.0,
        )
    )
    self.set_simulation_hotkeys_enabled(True)

    self.world = ke.sim.KangSimWorld(
        sim_dt=self.timing.physics_dt,
    )
```

Rates are the writable source of truth. `physics_dt`, `sim_dt`, `fixed_dt`,
and `decimation` are derived properties, so contradictory Hz and dt settings
cannot be supplied. General App loops may use fractional ratios because
`world.advance()` retains fractional time. Decimation-based RL environments
validate their integer decimation in their environment configuration.

## App timing VS runner policy

`SimulationTimingConfig` and `SimulationRunConfig` configure different loop
owners:

- A native `App` passes `SimulationTimingConfig` to `configure_timing()`.
  `App.start()` then schedules `fixedUpdate()` with its internal
  `FixedStepClock`. It does not accept or use `SimulationRunConfig`.
- An externally stepped runner, such as KELab, uses its timing configuration
  to determine `step_dt` and uses `SimulationRunConfig` independently to
  choose rendering and wall-clock pacing.

The connection in an external loop is the duration passed to the pacer:

```python
timing = ke.SimulationTimingConfig(
    physics_hz=120.0,
    fixed_update_hz=60.0,
)
run_config = ke.SimulationRunConfig(mode="paced")
pacer = ke.SimulationPacer(run_config)

world.advance(timing.fixed_dt)
pacer.wait(timing.fixed_dt)
```

`HEADLESS_FAST` disables rendering and wall-clock synchronization. `PACED`
waits only when an externally stepped simulation is ahead of wall time.
See [Simulation Run Modes](RUN_MODES.md) for KELab defaults and rendering
behavior.

Advance `KangSimWorld` by the callback duration instead of hard-coding a
physics substep count:

```python
def fixedUpdate(self, fixed_dt):
    self.world.advance(fixed_dt)

def preRender(self):
    self.visual.sync()
```

`world.advance()` does not change the PhysX timestep. It converts the requested
duration into whole `world.sim_dt` steps and retains any fractional remainder.
For example, a 120 Hz world performs two physics steps during one 60 Hz fixed
update and four during one 30 Hz fixed update.

Do not advance physics from `preRender()`. That couples simulation speed to
render frequency. Pure animation viewers may still update an interpolated
visual pose once per rendered frame because they do not integrate physics.

## Playback controls

Default simulation controls are opt-in:

```python
self.set_simulation_hotkeys_enabled(True)
```

- Enter or keypad Enter toggles play/pause.
- Space pauses a running simulation.
- Space advances one fixed update when already paused.
- `is_simulation_paused()` returns the authoritative state.

The same key meanings are used by KELab and its MimicKit-style viewer.

## Slow frames and catch-up

The fixed-step clock accumulates elapsed wall time. If a rendered frame is
late, the next frame may call `fixedUpdate()` multiple times. Configure its
safety limits with the same timing object:

```python
timing = ke.SimulationTimingConfig(
    max_catch_up_steps=8,
    max_frame_delta=0.25,
)
```

Work beyond that limit is discarded instead of retained as an ever-growing
backlog. `get_dropped_wall_time()` reports the total discarded time. A single
wall-clock delta is also clamped to 0.25 seconds, so a debugger stop or a
dragged window does not create an unbounded recovery burst.

At 60 Hz, the default limit can recover up to 133.3 ms of simulation per
rendered frame. At 120 Hz it covers 66.7 ms. The setting is a workload bound,
not a promise that simulation time always matches wall time.

The default of eight is intended for interactive applications:

- short and moderate stalls recover without changing simulation speed;
- severe sustained stalls slow simulation relative to wall time;
- physics work remains bounded, preventing a catch-up spiral.

For offline reinforcement learning or deterministic rollouts, do not schedule
environment steps from wall time. Drive environment or world steps explicitly.

## GPU stress test

Build the CUDA-enabled Python module and run:

```bash
make build_python_cuda

python/.venv/bin/python \
  python/examples/smoke/fixed_update_gpu_stress.py \
  --num-envs 64 \
  --frames 12 \
  --stall-ms 200 \
  --max-catch-up-steps 8
```

The test uses a PhysX GPU world and deliberately blocks `preRender()` to model
slow rendering. It verifies that per-frame updates never exceed the configured
limit and that GPU simulation time equals the number of executed fixed
updates.

Measurements on an RTX 4090, with a 60 Hz fixed update and 120 Hz physics:

| Render delay | Catch-up limit | Fixed updates | Dropped wall time | Simulated time |
|---:|---:|---:|---:|---:|
| 50 ms × 19 | 8 | 61 | 0.000 s | 1.017 s |
| 200 ms × 11 | 4 | 44 | 1.500 s | 0.733 s |
| 200 ms × 11 | 8 | 88 | ~0.8 s | 1.467 s |
| 200 ms × 11 | 16 | 139 | 0.000 s | 2.317 s |

With a 200 ms stall, limit 8 executed exactly eight fixed updates on every
delayed frame. Limit 16 followed wall time but required 12–13 updates per
frame. This supports keeping eight as the safe default while allowing
applications to raise it when wall-time tracking matters more than
responsiveness.

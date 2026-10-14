# Video Recording

KangEngine can record the framebuffer of an `App` to MP4. Recording captures
the rendered scene, while the red `REC` status shown in the viewer is not
included in the output.

By default, recordings use `libx264` through `imageio-ffmpeg`, are written to
`tmp/recordings`, and are capped at `1920x1080`. Larger framebuffers are scaled
before capture, while smaller framebuffers are not enlarged.
`imageio` and `imageio-ffmpeg` are included in the package dependencies.

## Viewer shortcuts

- `T`: save a PNG screenshot with a frame index and timestamp under `./tmp`.
- `Shift+T`: start or stop video recording.

## Configure recording

Configure the output directory, frame rate, or resolution before recording:

```python
import kangengine as ke


app: ke.App = ke.App()
app.set_video_recording_dir("captures")
app.set_video_recording_fps(60)
app.set_video_recording_resolution(1920, 1080)
```

The resolution is a maximum size, not an exact output size. Images are scaled
down proportionally to fit. With a cap enabled, each dimension is rounded down
to an even number even when no downscaling is needed.
With the recording cap set to `1920x1080`:

- Original framebuffer `1280x720` → video `1280x720`
- Original framebuffer `1281x721` → video `1280x720`
- Original framebuffer `1920x1080` → video `1920x1080`
- Original framebuffer `2560x1600` → video `1728x1080`

The original framebuffer size is unchanged.

Pass `None` to record at the native framebuffer resolution:

```python
app.set_video_recording_resolution(None)
```

`None` also disables even-dimension adjustment. Use an even-sized framebuffer
with the default codec; odd dimensions may cause an encoding error.

App recording selects video FPS in this order, using the first available value:

1. `app.start_video_recording(..., fps=...)` for this recording.
2. The value set by `app.set_video_recording_fps(...)`.
3. `SimulationTimingConfig.fixed_update_hz`, for offscreen recording with a
   timing configuration.
4. **60 FPS** otherwise.

Video FPS controls playback timing. Neither `render_hz` nor `physics_hz`
directly sets video FPS, and setting video FPS does not change those rates.

## Programmatic recording

```python
app.start_video_recording("captures/run.mp4")

# Render frames normally.

path = app.stop_video_recording()
```

`stop_video_recording()` finalizes the encoder and returns the output path.
Closing an `App` also finalizes an active recording.
Calling `start_video_recording()` while recording raises `RuntimeError`;
stop the current recording before starting another one.

An explicit output path is used as given, relative to the working directory
unless absolute. It is not appended to the configured recording directory.
Omitting the path creates a timestamped filename in that directory. Parent
directories are created automatically; an existing video at the chosen path is
overwritten. The recording directory does not affect screenshots.

Interactive recording follows wall-clock time. Frames are selected or repeated
as needed so that the video duration matches the elapsed recording time,
independently of the viewer render rate.

## Offscreen recording

`app.initialize(..., headless=True)` creates a renderer without a visible window.
`SimulationRunConfig(mode="headless_fast")` is a separate run policy: applying
it with `app.configure_run()` makes video recording start raise `RuntimeError`;
it does not remove the renderer or block explicit drawing. See
[Run modes](../simulation/RUN_MODES.md).

### Record 600 rendered frames

Render and capture 600 frames:

```python
app: ke.App = ke.App()
app.initialize(1280, 720, hide_ui=True, headless=True)
app.configure_timing(
    ke.SimulationTimingConfig(
        render_hz=0,
        physics_hz=120,
        fixed_update_hz=60,
    )
)
app.start_video_recording("captures/offscreen.mp4")

for _ in range(600):
    app.update_and_draw()

app.stop_video_recording()
```

`render_hz=0` disables App render-rate limiting; it is unrelated to resolution.
This manual loop does not use the rate limiting provided by `app.start()`.

Offscreen recording writes one video frame per rendered frame:
**video duration = recorded frame count / video FPS**. The loop above
therefore produces 600 frames at 60 FPS, or 10 seconds of video. Simulation
updates follow elapsed wall time, so this does not guarantee 600 simulation
steps or 10 seconds of simulated motion. Playback may be faster or slower than
the simulation's intended speed.

### Record 600 simulation steps

For explicit stepping, use a `ke.sim.KangSimWorld` named `world` configured with
`sim_dt=1.0 / 120.0`, and a `ke.visual.sim.SimWorldVisualizer` named `visual`
attached to the headless App above, with simulation objects already registered.
See [Rigid bodies](../simulation/RIGID_BODIES.md) and
[Visualization](../simulation/VISUALIZATION.md) for setup.
Here, `world.sim_dt` sets the physics timestep; the App timing configuration
does not schedule these explicit steps.

```python
app.set_video_recording_fps(120)
app.start_video_recording("captures/simulation_steps.mp4")

for _ in range(600):
    world.step(substeps=1)
    visual.sync()
    app.process_events()
    app.draw_frame()

app.stop_video_recording()
```

This records 600 physics steps and 600 frames: 5 seconds of simulation and
5 seconds of video, regardless of how long rendering takes. `process_events()`
handles window events and input. `draw_frame()` renders and records without
running App update callbacks.

## Record RGB frames directly

`VideoRecorder` can also encode `uint8` RGB arrays shaped
`[height, width, 3]` without an `App`. Its constructor's `fps` sets playback FPS
and defaults to **30 FPS**, independently of App settings:

```python
recorder: ke.recording.VideoRecorder = ke.recording.VideoRecorder(
    output_path="output.mp4",
    fps=60,
)

with recorder:
    recorder.write(frame=rgb_frame)
```

Pass an FFmpeg encoder name as `codec` to `VideoRecorder` (default: `libx264`).
Available encoders depend on the FFmpeg binary used by `imageio-ffmpeg`.
`VideoRecorder` passes input dimensions to the encoder unchanged; the App's
resolution cap and even-dimension adjustment do not apply. Use even widths and
heights with the default codec; odd dimensions may cause an encoding error.
All frames in one recording must have the same resolution.

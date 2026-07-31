# Video Recording

KangEngine can record the framebuffer of an `App` to MP4. Recording captures
the rendered scene, while the red `REC` status shown in the viewer is not
included in the output.

By default, recordings use `libx264` through `imageio-ffmpeg`, are written to
`tmp/recordings`, and are capped at `1920x1080`. Larger framebuffers are scaled
before capture, while smaller framebuffers are not enlarged.

## Viewer shortcuts

- `T`: save one screenshot.
- `Shift+T`: start or stop video recording.

## Configure recording

Configure the output directory, frame rate, or resolution before recording:

```python
import kangengine as ke


app = ke.App()
app.set_video_recording_dir("captures")
app.set_video_recording_fps(60)
app.set_video_recording_resolution(1920, 1080)
```

Pass `None` to record at the native framebuffer resolution:

```python
app.set_video_recording_resolution(None)
```

## Programmatic recording

```python
app.start_video_recording("captures/run.mp4")

# Render frames normally.

path = app.stop_video_recording()
```

`stop_video_recording()` finalizes the encoder and returns the output path.
Closing an `App` also finalizes an active recording.

Interactive recording follows wall-clock time. Frames are selected or repeated
as needed so that the video duration matches the elapsed recording time,
independently of the viewer render rate.

## Offscreen recording

An App initialized with `headless=True` uses offscreen rendering:

```python
app = ke.App()
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
    app.render_frame_once()

app.stop_video_recording()
```

Offscreen recording writes one video frame for each rendered frame and uses
`fixed_update_hz` as its default video FPS. `HEADLESS_FAST` creates no renderer
and therefore cannot record video.

## Record RGB frames directly

`VideoRecorder` can also encode `uint8` RGB arrays shaped
`[height, width, 3]` without an `App`:

```python
recorder = ke.recording.VideoRecorder("output.mp4", fps=60)

with recorder:
    recorder.write(rgb_frame)
```

Pass `codec` to `VideoRecorder` to select another available encoder.
All frames in one recording must have the same resolution.

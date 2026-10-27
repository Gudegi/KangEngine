# Scene editing, capture and recording

[Start and connect](../MCP.md) before using these workflows.

## Build a scene with text

After starting `serve` and connecting your AI client, first create the scene:

> Create a gray floor six units across, a red unit box, and a blue unit sphere.
> Place the box center half a unit above the floor and the sphere two units to
> its right. Frame the box with the camera and show me a capture.

Then edit and record it:

> Make the box green and twice as tall, with its center one unit above the floor.
> Frame it again and show me the result. Record demo.mp4 while showing the scene
> from a few different viewpoints, stop recording, then delete the sphere.

Deletion may require client approval. Wait for each edit to succeed before
requesting `viewport_capture`; do not call dependent tools in parallel.
`camera_set` changes views immediately, so repeated calls do not produce a smooth orbit.

Creation accepts `box`, `sphere` and `plane`. The box side, sphere diameter and
plane side are one scene unit before scaling. A plane's normal follows the App
up axis. Paths such as `/floor` and `/box` must be new; parents must already
exist as root/Xform prims. Path segments use letters, digits and underscores,
starting with a letter or underscore. Objects are static visuals without
collision bodies. Object and material capacities are in the {ref}`limits <mcp-limits>`.

Transform edits preserve omitted or null fields. Supply at least one of
`translation`, `rotation_wxyz` or `scale`; each supplied value is absolute
and parent-relative, not an increment. Translation uses scene units, rotation
is a normalized wxyz quaternion and scale is positive.
Transform editing supports leaves only. Physics/ExternalBuffer objects and their
descendants are rejected; application callbacks may still overwrite edits to
application-owned static objects.

```{note}
For shear, reflection or singular transforms, see
{ref}`transform edge cases <mcp-transform-details>`.
```

`scene_set_material` assigns an untextured Phong material using linear RGB and
shininess. It resets the object's color multiplier to white; existing texture
overrides and alpha mode remain. Other objects are unaffected.

`camera_focus` automatically fits automation-created primitives while preserving
the viewing direction. For other static prims, supply `distance` to target their
world origin. It does not reveal hidden objects.
`scene_delete` only deletes childless objects created through automation in this
session. These tools do not provide import, hierarchy authoring, physics editing,
scene persistence or undo.

## Measure a fixed number of frames

```bash
python -m kangengine.adapters.mcp profile-start --frame-count 60
python -m kangengine.adapters.mcp profile-status JOB_ID
python -m kangengine.adapters.mcp profile-result JOB_ID --max-frames 30
python -m kangengine.adapters.mcp profile-result JOB_ID --offset 30 --max-frames 30
```

Replace `JOB_ID` with the returned ID. Read results after `state` becomes
`completed`, `failed` or `cancelled`; use `next_offset` for the next page.
`profile-cancel JOB_ID` stops measurement. The previous profiler setting is restored
when the job ends; do not change it manually during a job.

Results freeze at job end; GPU samples may remain `PENDING` with `null` durations.
Use `renderer_profile` for later revisions in renderer history. Avoid captures,
recording and repeated queries during measurement because they add overhead.

## Adjust the view and visibility

```bash
python -m kangengine.adapters.mcp camera
python -m kangengine.adapters.mcp camera-set --position 4 3 4 --target 0 0 0
python -m kangengine.adapters.mcp visibility /subject false
python -m kangengine.adapters.mcp visibility /subject true
```

Replace `/subject` with a scene path. Camera coordinates are world-space scene
units. Clear any active scene camera before controlling the navigation camera;
App callbacks can still move it afterward.

Visibility edits leave `active` and child flags unchanged. Hidden/inactive parents
still suppress rendering; ExternalBuffer visibility affects the whole batch.
Camera and visibility changes persist until explicitly changed again.

## Record video

```bash
python -m kangengine.adapters.mcp recording-start --output take-01.mp4 --fps 30
python -m kangengine.adapters.mcp recording-status
python -m kangengine.adapters.mcp recording-stop
```

PNGs default to `tmp/captures` and MP4s to `tmp/recordings`, relative to the
**App's** working directory. Choose distinct filenames when Apps share a directory.
For remote Apps, files stay on the remote machine; copy saved videos separately.
PNG content is also returned to the client.

Use a new filename within the allowed directory; see {ref}`limits <mcp-limits>`.
Recording requires rendering and at least one completed frame. It uses the App's
resolution cap: paced mode follows wall time; offscreen-fast records every rendered
frame. Application UI is excluded. A static scene and camera produce a still video.

Recording continues after client disconnection; stop it or close the App to
finalize the file. MCP shares the UI/Python recorder: starting an active recorder
returns `BUSY`, and `recording_stop` also stops recordings started through those interfaces.

(mcp-app-integration)=
## Integrate MCP with a manually stepped App

Enable MCP explicitly and keep the frame loop running to service requests:

```python
import kangengine as ke

app: ke.App = ke.App()
try:
    app.initialize(width=640, height=480)
    app.enable_mcp(port=8765)
    while not app.should_close():
        app.update_and_draw()
finally:
    app.close()
```

Call `setup()` once before the loop for subclasses that need it. Keep creation,
rendering and cleanup on the App thread; call `close()` in `finally`.
`process_events()` alone does not service MCP requests. Do not wait on a pending
capture inside a render callback.

For custom output directories, set `app.automation.output_dir` (PNG) and
`app.automation.recording_dir` (MP4) to resolved `Path` values before requests arrive.
`headless=True` hides the window but still requires a display and OpenGL context;
`HEADLESS_FAST` disables capture.

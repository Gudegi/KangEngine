# MCP tool reference

[Start and connect](../MCP.md) · [Workflows](WORKFLOWS.md)

(mcp-configuration)=
## Configuration

| Setting | Meaning |
| --- | --- |
| `KANGENGINE_MCP=1` | Enable MCP at `App.start()`; off by default. |
| `KANGENGINE_MCP_PORT` | Port for the environment opt-in; default 8765. For `serve`, use `--port`. |
| `KANGENGINE_MCP_TOKEN` | Supplied credential; otherwise generated at each server start. |
| `KANGENGINE_MCP_TOOLS` | Comma-separated tool groups; replaces the defaults. |

One active App per process is supported. For multiple Apps, use separate processes
and ports: `serve --port 8766` or `serve --port 0` for an available port.

### Tool groups

Default groups are `scene,camera,capture,recording,profile,docs`: **24 tools
including `engine_status`**. Without guide files, `search_docs` and `get_doc` are
omitted at startup, leaving **22 tools**; `get_api_docs` remains available.
The optional `scenario` group adds five tools and requires registered MCP targets
for playback. See the **Group** column below for membership.

```bash
KANGENGINE_MCP_TOOLS=scene,camera,capture python -m kangengine.adapters.mcp serve
```

Unknown or empty group selections fail at startup. Restart the server and reconnect
the client after changing groups. `engine_status` is always available; its
`available_tools` lists exposed tools, while capabilities describe runtime support.
Groups are not user roles or an authorization system.

(mcp-token-reuse)=
### Reuse a token

To retain the current App's generated token, run this while it is still running,
in the shell that will launch its next instance:

```bash
export KANGENGINE_MCP_TOKEN="$(python -m kangengine.adapters.mcp token)"
```

Close the old App, then launch `serve` from that shell. Supplied tokens require
at least 32 printable ASCII characters without spaces; use a generated token
rather than a password. Store it privately if it must survive shell restarts.

To generate a new token, run `unset KANGENGINE_MCP_TOKEN` in the App-launching
shell before restarting the App, then refresh the client's credential and restart
the client or its IDE host process.

(mcp-client-config)=
### Client configuration output

`print-config` prints settings for the selected App without revealing its token
or editing client configuration. On the default port, its JSON output is:

```json
{
  "transport": "streamable-http",
  "url": "http://127.0.0.1:8765/mcp",
  "headers": {
    "Authorization": "Bearer <token>"
  },
  "token_command": "python -m kangengine.adapters.mcp token"
}
```

Replace `<token>` with the output of the `token` command. With `--format codex`:

```toml
[mcp_servers.kangengine]
url = "http://127.0.0.1:8765/mcp"
bearer_token_env_var = "KANGENGINE_MCP_TOKEN"
```

See the [official Codex MCP guide](https://developers.openai.com/codex/mcp)
for client configuration.

## Tools

Exposure follows the {ref}`configuration <mcp-configuration>`.
`Always available` is not a configurable group name.

| Tool | Group | Inputs | Result |
| --- | --- | --- | --- |
| `engine_status` | Always available | None | Session, readiness, backends, coordinates, capabilities and limits. Refreshed at a frame boundary once ready. |
| `scene_inspect` | `scene` | `path="/"`, `max_depth=3`, `max_prims=200` | Copied hierarchy, transforms, active/visible state and resource IDs. |
| `scene_create_primitive` | `scene` | `path`, `shape`, optional `translation`, `rotation_wxyz`, `scale`, `color` | Create a box, sphere or plane with a registered mesh and material. |
| `scene_set_transform` | `scene` | `path`, at least one of `translation`, `rotation_wxyz`, `scale` | Edit a static leaf's local transform; omitted/null fields stay unchanged. |
| `scene_set_material` | `scene` | `path`, `color`, `shininess=32` | Assign an untextured Phong material to one static visual. |
| `scene_delete` | `scene` | `path` | Delete a childless object created through automation in this App session. |
| `object_inspect` | `scene` | `path` | One prim's transforms, resource IDs, render flags and physics metadata. |
| `resource_inspect` | `scene` | `handle`, `max_paths=100` | Resource type/name/URI, usage count and bounded usage paths. |
| `camera_focus` | `camera` | `path`, optional `distance` | Auto-fit an automation-created primitive; explicit distance targets a static prim's world origin. |
| `search_docs` | `docs` | `query`, `max_results=10` | Literal text search in the source checkout's public guide. |
| `get_doc` | `docs` | `path`, `start_line=1`, `max_lines=80` | Read a bounded portion of a public guide page. |
| `get_api_docs` | `docs` | `symbol` | Public symbol signature and docstring, without executing code. |
| `viewport_capture` | `capture` | Optional `output_path` | PNG image content and dimensions, native frame ID and optional saved path. |
| `renderer_profile` | `profile` | `max_frames=60` | Existing profile snapshots and summary; never enables profiling. |
| `profile_start` | `profile` | `frame_count=60` | Measure a bounded number of normal frames; return a job ID. |
| `profile_status` | `profile` | `job_id` | Measurement progress and previous profiler setting. |
| `profile_result` | `profile` | `job_id`, `offset=0`, `max_frames=30` | Copied frame results, with pagination. |
| `profile_cancel` | `profile` | `job_id` | Stop measurement and restore the previous profiler setting. |
| `camera_get` | `camera` | None | Navigation camera position/target in world coordinates and active scene camera path. |
| `camera_set` | `camera` | `position`, `target` | Set navigation camera position and target, each three scene-unit values. |
| `scene_set_visibility` | `scene` | `path`, `visible` | Set one prim's local visibility; report effective visibility. |
| `recording_status` | `recording` | None | Common App recorder state and current or last output path. |
| `recording_start` | `recording` | `output_path`, `fps=30` | Start recording to a new MP4 within the allowed output directory. |
| `recording_stop` | `recording` | None | Finalize the common App recording. |
| `scenario_targets` | `scenario` | None | Registered targets, supported scenarios and data/configuration identifiers. |
| `scenario_start` | `scenario` | `target`, `scenario`, `duration`, optional `output_path` | Reset the target and start a fixed-step job; return its ID. |
| `scenario_status` | `scenario` | `job_id` | Progress, terminal state, pose snapshot and capture metadata. |
| `scenario_cancel` | `scenario` | `job_id` | Stop at a step boundary and release live control. |
| `scenario_capture` | `scenario` | `job_id` | Return a completed job's retained PNG without rendering again. |

Matrices are serialized by rows; positions use scene units and the reported up
axis. `meters_per_unit` is `null`. Resource-only prims have no transforms;
ExternalBuffer matrices describe the batch owner, not individual instances.
Large geometry, texture and tensor buffers are omitted.

PNGs contain the postprocessed RGB scene with a top-left origin and no UI, at
the presented framebuffer's dimensions. They are returned to the client and saved
only when requested. Cancellation cannot undo a file commit already in progress;
App shutdown waits for it to finish.

The App must keep rendering to answer requests. Capture and large queries can
increase frame time. Profiler `enabled` and `data_status` distinguish disabled,
empty and retained results; unavailable timings are `null` without waiting for the GPU.

Authored materials remain retained until the App closes, including after object
deletion, and count toward the {ref}`limits <mcp-limits>`.

(mcp-transform-details)=
### Transform edge cases

Translation-only edits preserve shear/reflection. Changing only rotation or only
scale requires a nonsingular, non-reflected TRS matrix; otherwise supply both to
replace the linear part.

## Common response fields

Fields below describe successful responses; documentation tools do not carry an
App session ID. Frame IDs are zero-based native frame indices, comparable only
within the same `app_session_id`, and share the profiler's `frame_index` numbering.

| Field | Returned by | Meaning |
| --- | --- | --- |
| `app_session_id` | All non-documentation tools | Identifies the App automation session, not the HTTP connection. |
| `frame_id` | `viewport_capture`, `scenario_capture`; nested in completed scenario job `capture` metadata | The rendered frame used for the PNG. Retained scenario captures keep their original ID. |
| `observed_before_frame_id` | `scene_inspect`, `renderer_profile` | Observation was taken before this frame's updates/render. It does not identify every frame in a profile history result. |
| `processed_before_frame_id` | Scene tools except `scene_inspect`; all camera and recording tools; `profile_start/status/result/cancel`; `scenario_targets/start/status/cancel` | Command was processed before this frame's updates/render. For job queries this is the query boundary, not the job's measured/captured frame. |
| `last_frame_id` | `engine_status` | Last completed frame at the status observation; `null` before the first frame. |
| `job_id` | `profile_start/status/result/cancel`, `scenario_start/status/cancel` | Opaque handle scoped to that job type and App session; invalid after eviction or App restart. |
| `truncated` | `scene_inspect`, `resource_inspect`, `search_docs`, `get_doc`, `get_api_docs` | Content was omitted due to the query or output bound. Narrow the query or use the tool's paging controls where available. |
| `next_offset` | `profile_result` | Pass as the next `offset`; `null` when no more retained job frames remain. |

## Ordering edits and captures

Wait for an edit's successful response before calling `viewport_capture`.
Its frame ID is the edit's `processed_before_frame_id` or later in the same session.
Parallel calls do not guarantee this order. Callbacks and animation may change
the scene before capture. Server instructions and the capture tool description
also specify this rule.

## CLI mapping

Run commands as `python -m kangengine.adapters.mcp COMMAND`. Put
`--connection-file PATH` before the command when selecting a session.

| CLI command | MCP tool / action |
| --- | --- |
| `serve` | Open a new blank App |
| `sessions` | List local records and process state; no token output |
| `token` | Print the credential from the selected connection file, not `KANGENGINE_MCP_TOKEN` |
| `print-config [--format codex]` | Print generic JSON or Codex TOML without credentials |
| `status` | `engine_status` |
| `scene` | `scene_inspect` |
| `capture` | `viewport_capture` |
| `profile` | `renderer_profile`: read existing snapshots |
| `profile-start/status/result/cancel` | `profile_start/status/result/cancel` |
| `camera`, `camera-set` | `camera_get`, `camera_set` |
| `visibility` | `scene_set_visibility` |
| `recording-start/status/stop` | `recording_start/status/stop` |
| `targets`, `scenario` | `scenario_targets`, `scenario_start` (optional group) |
| `job-status/cancel/capture` | `scenario_status/cancel/capture` (optional group) |

Other tools are available directly to MCP clients. `frame_count` selects how many
frames to measure; `max_frames` limits how many a query returns.

(mcp-limits)=
## Limits

| Operation | Limit |
| --- | --- |
| Outstanding requests / processing | 16 / 4 per frame |
| Request deadline | 10 seconds including queue time; not the whole profile/scenario job |
| HTTP request / tool JSON response | 64 KiB / 1 MiB |
| Capture | 16,777,216 pixels; PNG at most 8 MiB; at most 2 encoding jobs |
| PNG output | New `.png` within the service's capture directory (default `tmp/captures`); existing files are rejected |
| Scene inspection | Depth 0..16; 1..1000 prims; resource usage query 1..1000 paths |
| Primitive authoring | 256 live objects; 128 distinct retained materials per App |
| Supplied translation / scale | ±1,000,000 per axis / 0.0001..10000 per axis |
| Material | Linear RGB 0..1; shininess 1..256 |
| Explicit focus distance | 0.001..1,000,000 scene units |
| Recording | 1..240 fps; new `.mp4` within the service's recording directory (default `tmp/recordings`); existing files are rejected |
| Profiling | 1..240 measured frames; result pages 1..60 frames; 1 active / 4 retained jobs |
| Existing profile query | 1..240 frames |
| Scenario (optional) | 0..120 logical seconds; 1 active / 4 retained jobs; 8 steps per frame with a 4 ms scheduling budget |
| Documentation | Search up to 30 results; read up to 120 lines per call |

(mcp-errors)=
## Errors and recovery

Execution errors return `is_error=true` and
`{"error": {"code": "...", "message": "..."}}`.
Protocol/schema errors remain SDK errors. Read `message` for the specific cause.

| Code / symptom | What to do |
| --- | --- |
| `INVALID_ARGUMENT` | Fix inputs; use a new filename and keep output within its allowed directory. |
| `NOT_FOUND` | Inspect current paths or job IDs; an old job may have been evicted. |
| `UNSUPPORTED` | Check ownership, rendering mode and feature scope. If the guide directory disappeared after startup, use `get_api_docs`. |
| `NOT_READY` | Wait for the first render or job completion. |
| `BUSY` | Wait for pending work; finish the active job/recording or respect resource limits. |
| `TIMEOUT` | Confirm the App is still producing frames. Check current state before retrying a mutation. |
| `CANCELLED` | Request was canceled; inspect current state before retrying. |
| `CLOSED` | Start/connect to a running App session. |
| `CAPTURE_FAILED` | Check rendering, destination permissions and the App's error output. |
| `OUTPUT_TOO_LARGE` | Request fewer prims/frames/lines or reduce capture resolution. |
| `INTERNAL_ERROR` | Inspect the App error and reproduce with a bounded request. |
| Connection refused / HTTP 401 | Check the running App's URL and current credential using `sessions` and `token`. |
| Multiple session candidates | Select one with `--connection-file PATH`; `unknown` process state is not assumed dead. |
| Missing tool | Check `KANGENGINE_MCP_TOOLS` and `available_tools`. Guide search/read tools are omitted without guide files; `get_api_docs` still works. Restart the server and reconnect after changing groups or installing the guide. |
| Client blocks deletion | Use the client's approval workflow; tool annotations are hints, not permission grants. |

`engine_status` may report `initializing` before the first frame. A hidden GLFW
window still requires a display; `headless=True` is not a displayless renderer.

### Stale or ambiguous sessions

Automatic selection excludes `dead` records and requires one remaining candidate;
`unknown` records remain candidates. Use `--connection-file PATH` if ambiguous.
Normal shutdown removes the App's record; stale records are not automatically deleted.

`alive` is a process check, not a connection test, and cannot detect PID reuse on
macOS. Use `status` to verify connectivity and the App session ID.

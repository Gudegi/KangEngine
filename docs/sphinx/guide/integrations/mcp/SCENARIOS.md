# Optional MCP scenario jobs

[Start and connect](../MCP.md) · [Tool reference](REFERENCE.md)

## Capture a specified playback time

Scenario jobs require an App with a registered MCP target. Ordinary scene editing,
capture and recording do not. Enable the optional `scenario` group at startup:

```bash
KANGENGINE_MCP=1 KANGENGINE_MCP_TOOLS=scene,camera,capture,recording,profile,docs,scenario python path/to/your_app.py
```

Run `targets` to find supported targets and scenarios; enabling the group does
not create them. The example below assumes target `fixture` and scenario `orbit`:

```bash
python -m kangengine.adapters.mcp targets
python -m kangengine.adapters.mcp scenario fixture --scenario orbit --duration 13 --output at-13s.png
python -m kangengine.adapters.mcp job-status JOB_ID
python -m kangengine.adapters.mcp job-capture JOB_ID
```

Replace `JOB_ID` with the returned `job_id`. Wait for `completed` before retrieving
the capture; `failed` and `cancelled` also end the job. Client disconnection does
not stop it. To cancel:

```bash
python -m kangengine.adapters.mcp job-cancel JOB_ID
```

Each job **resets and changes the selected target's playback state**. `duration`
is logical seconds after reset, not wall-clock waiting time. Steps are 1/60 s,
with a shorter final step if needed; zero captures the reset pose.
The previous pose is **not restored** when the job ends.

Use `job-capture` (`scenario_capture`) for the retained frame: playback may resume
after completion, so a later `viewport_capture` can show a different pose.
Cancellation does not undo a PNG save already being committed.

The App must keep rendering to advance or cancel jobs; `HEADLESS_FAST` is
unsupported. Closing the App stops unfinished jobs. Results follow the
{ref}`retention limits <mcp-limits>`; evicted IDs return `NOT_FOUND`, but saved files
remain. Reproducibility depends on the target and its inputs.

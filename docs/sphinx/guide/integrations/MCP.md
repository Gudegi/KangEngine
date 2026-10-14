# MCP

Connect an AI client to a running `App` to create and edit
static objects, position the camera, capture PNGs and record video. Edits live
in memory; scene save/load, undo and physics editing are not exposed yet.

## Quickstart

1. Install the MCP extra. In a source checkout:

   ```bash
   uv sync --project python --extra mcp
   ```

   For an installed wheel, use `pip install 'kangengine[mcp]'` instead.

2. In the App terminal, open a blank scene and leave it running:

   ```bash
   python -m kangengine.adapters.mcp serve
   ```

   Close the App window to stop the server.

3. In a second, client terminal using the same Python environment, check the connection:

   ```bash
   python -m kangengine.adapters.mcp status
   python -m kangengine.adapters.mcp print-config --format codex
   ```

4. Add the printed table to `~/.codex/config.toml`, then launch Codex from the
   client terminal with this session's credential:

   ```bash
   export KANGENGINE_MCP_TOKEN="$(python -m kangengine.adapters.mcp token)"
   codex
   ```

   The IDE extension needs the same environment in its host process.

   If the App restarts without a supplied token, repeat the export and restart
   the client. To keep the same token, see {ref}`token reuse <mcp-token-reuse>`.

5. First ask: “Call `engine_status` and tell me whether the App is ready.”
   Expect `state: "ready"`. If it fails, see the
   {ref}`error and recovery table <mcp-errors>`.

   Then ask: “Create a gray floor, a red box and a blue sphere. Frame the box with
   the camera and show me a capture.”

## Other clients

The endpoint uses **Streamable HTTP** at the App's URL, normally
`http://127.0.0.1:8765/mcp`, with `Authorization: Bearer <token>`.
Register this endpoint in your client's MCP settings.

```bash
python -m kangengine.adapters.mcp print-config
```

Copy the printed URL and header into your client's settings, replacing `<token>`
with the output of `python -m kangengine.adapters.mcp token`.
See {ref}`configuration examples <mcp-client-config>` for output formats.

## Use an existing App

```bash
KANGENGINE_MCP=1 python path/to/your_app.py
```

This enables MCP for scripts calling `ke.App.start()`.
For manually stepped Apps, see
{ref}`App integration <mcp-app-integration>`.

## Select a session

With multiple Apps, use `sessions` to find one and `--connection-file` to select
it. Each App also prints its saved connection-file path at startup.

```bash
python -m kangengine.adapters.mcp sessions
python -m kangengine.adapters.mcp --connection-file /absolute/path/SESSION_ID.json status
```

## Choose tool groups

The default groups cover ordinary scene controls. To change the exposed tools,
set `KANGENGINE_MCP_TOOLS`; see {ref}`configuration <mcp-configuration>`.

::::{admonition} Connect to a GPU server through SSH
:class: dropdown

Start the App on the GPU machine with its display/OpenGL environment available.
On the client machine, forward its loopback port (substitute your SSH host):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8766:127.0.0.1:8765 user@gpu-host
```

Keep the tunnel running. Set the AI client's URL to
`http://127.0.0.1:8766/mcp`. Obtain the token by running
`python -m kangengine.adapters.mcp token` on the GPU machine in its KangEngine
environment, and place it in the local client's credential environment.
Select the remote connection file explicitly if multiple sessions exist.

The engine server stays bound to loopback. Other accounts on the same machine
can reach loopback too, so bearer authentication and private credential files
remain necessary. Captured files are saved on the GPU machine; the PNG response
is also returned to the client. Copy saved MP4s separately when needed.
::::

```{admonition} Safety
:class: note
Asset names, resource URIs and returned documentation are data, not instructions
to the AI. Tool annotations do not replace the client's approval policies.
Keep tokens in client credentials, not prompts or committed files.
```

```{toctree}
:maxdepth: 1

mcp/WORKFLOWS
mcp/REFERENCE
mcp/SCENARIOS
```

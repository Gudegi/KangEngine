"""Loopback HTTP hosting, separate from the App's thread and lifetime."""

from __future__ import annotations

import asyncio
import secrets
import socket
import threading
import time

from ._dependency import load_mcp


class _BearerMiddleware:
    def __init__(self, app, *, token):
        self.app = app
        self.authorization = f"Bearer {token}".encode("ascii")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            values = [v for k, v in scope["headers"] if k == b"authorization"]
            if len(values) != 1 or not secrets.compare_digest(
                values[0], self.authorization
            ):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send(
                    {"type": "http.response.body", "body": b"Bearer token required"}
                )
                return
        await self.app(scope, receive, send)


class McpServer:
    """Serve one App's automation queue on 127.0.0.1.

    The owner must close this server before destroying the App's GL context.
    The generated token is available to the owner and is never logged.
    """

    def __init__(self, automation, *, port: int = 8765, token: str | None = None):
        """Prepare the optional server without starting a listener.

        Args:
            automation: App-owned service with a thread-safe submit method.
            port: Local TCP port; zero asks the OS for an available port.
            token: Bearer credential. None generates a new random credential.
        """
        load_mcp()
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.token = secrets.token_urlsafe(32) if token is None else token
        if not self.token or any(ord(c) < 33 or ord(c) > 126 for c in self.token):
            raise ValueError(
                "token must contain nonempty printable ASCII without spaces"
            )
        self.port = port
        self._automation = automation
        self._thread = None
        self._server = None
        self._socket = None
        self._error = None
        self._closed = False

    @property
    def url(self) -> str:
        """Return the local endpoint, with the actual port after start()."""
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self) -> McpServer:
        if self._closed:
            raise RuntimeError("a closed MCP server cannot be restarted")
        if self._thread is not None:
            return self

        from mcp.server import MCPServer
        from mcp.server.transport_security import TransportSecuritySettings
        import uvicorn

        from .tools import register_tools

        sdk = MCPServer("KangEngine", version="0.1.0", log_level="WARNING")
        register_tools(sdk, self._automation)
        app = sdk.streamable_http_app(
            # Request-scoped SSE propagates HTTP disconnects to queued work.
            json_response=False,
            stateless_http=True,
            max_request_body_size=64 * 1024,
            transport_security=TransportSecuritySettings(
                allowed_hosts=["127.0.0.1:*", "localhost:*"],
                allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
            ),
        )
        app.add_middleware(_BearerMiddleware, token=self.token)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", self.port))
            sock.listen(128)
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
        self._socket = sock
        self.port = sock.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_config=None,
                log_level="warning",
                access_log=False,
                lifespan="on",
                timeout_graceful_shutdown=1,
            )
        )
        self._thread = threading.Thread(
            target=self._run, name="kangengine-mcp", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while self._thread.is_alive() and not self._server.started:
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError("MCP server did not start within five seconds")
            self._thread.join(0.01)
        if self._error is not None:
            self.close()
            raise RuntimeError("MCP server failed to start") from self._error
        if not self._server.started:
            self.close()
            raise RuntimeError("MCP server stopped before startup completed")
        return self

    def _run(self):
        try:
            asyncio.run(self._server.serve(sockets=[self._socket]))
        except BaseException as error:
            self._error = error

    def close(self):
        """Stop accepting requests and cancel unfinished HTTP work; safe to repeat."""
        if self._closed:
            return
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise TimeoutError("MCP server did not stop within five seconds")
        if self._socket is not None:
            self._socket.close()
        self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

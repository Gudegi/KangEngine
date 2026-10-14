"""Per-user connection records shared by the App and local MCP clients."""

import json
import os
from pathlib import Path
import sys
import tempfile


def _absolute_env_path(name):
    value = os.environ.get(name)
    path = Path(value) if value else None
    return path if path is not None and path.is_absolute() else None


def connection_directory() -> Path:
    """Return a CWD-independent Linux/macOS directory without creating it."""
    if sys.platform == "darwin":
        base = Path.home() / "Library/Caches"
    else:
        base = (
            _absolute_env_path("XDG_RUNTIME_DIR")
            or _absolute_env_path("XDG_CACHE_HOME")
            or Path.home() / ".cache"
        )
    return base / "kangengine" / "mcp"


def _process_start(pid):
    """Linux start ticks distinguish a reused PID without probing the endpoint."""
    if sys.platform != "linux":
        return None
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def process_state(record):
    pid = record.get("pid")
    if type(pid) is not int or not 0 < pid <= 2**31 - 1:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "unknown"
    saved_start = record.get("process_start")
    current_start = _process_start(pid)
    if (
        saved_start is not None
        and current_start is not None
        and saved_start != current_start
    ):
        return "dead"
    return "alive"


def write_connection(*, url, token, app_session_id) -> Path:
    """Atomically publish one App's credentials; POSIX files use mode 0600."""
    directory = connection_directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise PermissionError(
            "MCP connection directory must be owned by this user and not a symlink"
        )
    directory.chmod(0o700)
    path = directory / f"{app_session_id}.json"
    descriptor, temporary_name = tempfile.mkstemp(dir=directory, suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "url": url,
                    "token": token,
                    "app_session_id": app_session_id,
                    "pid": os.getpid(),
                    "process_start": _process_start(os.getpid()),
                },
                stream,
            )
            stream.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def remove_connection(path: Path, *, app_session_id):
    """Remove only the closing App's record, preserving a replacement record."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    if record["app_session_id"] == app_session_id:
        path.unlink(missing_ok=True)


def connection_files() -> list[Path]:
    return sorted(connection_directory().glob("*.json"))


def select_connection_file(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    files = [
        Path(session["connection_file"])
        for session in list_connections()["sessions"]
        if session["process_state"] != "dead"
    ]
    if not files:
        raise FileNotFoundError(
            f"No live MCP connection candidates in {connection_directory()}; start an App "
            "with MCP enabled, or inspect 'sessions' for stale records"
        )
    if len(files) > 1:
        raise ValueError(
            "Multiple MCP connection files found; run 'python -m kangengine.adapters.mcp sessions' "
            "and select one with --connection-file PATH"
        )
    return files[0]


def list_connections() -> dict:
    """List saved metadata and local process state without revealing credentials."""
    sessions = []
    for path in connection_files():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # An App can finish between discovery and reading its record.
            continue
        sessions.append(
            {
                "connection_file": str(path),
                "url": record["url"],
                "app_session_id": record["app_session_id"],
                "pid": record.get("pid"),
                "process_state": process_state(record),
            }
        )
    return {"directory": str(connection_directory()), "sessions": sessions}

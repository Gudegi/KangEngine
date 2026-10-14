"""Session discovery is independent of CWD and never lists bearer credentials."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kangengine.adapters.mcp import _connection as connection


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(connection.sys, "platform", "linux")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    return tmp_path / "runtime" / "kangengine" / "mcp"


@pytest.mark.parametrize(
    "platform,variables,suffix",
    [
        ("linux", {"XDG_RUNTIME_DIR": "/runtime"}, "/runtime/kangengine/mcp"),
        (
            "linux",
            {"XDG_RUNTIME_DIR": "relative", "XDG_CACHE_HOME": "/cache"},
            "/cache/kangengine/mcp",
        ),
        (
            "linux",
            {"XDG_RUNTIME_DIR": "", "XDG_CACHE_HOME": "relative"},
            ".cache/kangengine/mcp",
        ),
        ("darwin", {}, "Library/Caches/kangengine/mcp"),
    ],
)
def test_platform_paths_do_not_depend_on_cwd(
    runtime, monkeypatch, tmp_path, platform, variables, suffix
):
    monkeypatch.setattr(connection.sys, "platform", platform)
    for key, value in variables.items():
        if value.startswith("/"):
            value = str(tmp_path / value.lstrip("/"))
        monkeypatch.setenv(key, value)
    expected = (
        tmp_path / suffix.lstrip("/")
        if suffix.startswith("/")
        else tmp_path / "home" / suffix
    )
    before = connection.connection_directory()
    monkeypatch.chdir(tmp_path)
    assert connection.connection_directory() == before == expected


def test_multiple_sessions_selection_permissions_and_own_cleanup(
    runtime, monkeypatch, tmp_path
):
    with pytest.raises(FileNotFoundError, match="start an App"):
        connection.select_connection_file()
    first = connection.write_connection(
        url="http://127.0.0.1:1001/mcp", token="private-one", app_session_id="one"
    )
    monkeypatch.chdir(tmp_path)
    assert connection.select_connection_file() == first
    second = connection.write_connection(
        url="http://127.0.0.1:1002/mcp", token="private-two", app_session_id="two"
    )
    assert first.stat().st_mode & 0o777 == 0o600
    assert runtime.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError, match="--connection-file"):
        connection.select_connection_file()
    assert connection.select_connection_file(first) == first
    listing = connection.list_connections()
    assert len(listing["sessions"]) == 2
    assert listing["sessions"][0]["pid"] == os.getpid()
    assert "private-" not in json.dumps(listing) and '"token"' not in json.dumps(
        listing
    )
    connection.remove_connection(first, app_session_id="wrong")
    assert first.exists()
    connection.remove_connection(first, app_session_id="one")
    connection.remove_connection(first, app_session_id="one")
    assert not first.exists() and connection.select_connection_file() == second
    connection.remove_connection(second, app_session_id="two")
    assert connection.list_connections()["sessions"] == []


def test_failed_publication_cleans_temporary_credentials(runtime, monkeypatch):
    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        connection.write_connection(
            url="http://localhost:1001/mcp", token="secret", app_session_id="one"
        )
    assert list(runtime.iterdir()) == []


def test_symlink_directory_is_not_used_for_credentials(runtime, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    runtime.parent.mkdir(parents=True)
    runtime.symlink_to(target, target_is_directory=True)
    with pytest.raises(PermissionError):
        connection.write_connection(
            url="http://localhost:1001/mcp", token="secret", app_session_id="one"
        )
    assert not list(target.iterdir())


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Subprocess home isolation is not configured for macOS",
)
def test_sessions_cli_is_cwd_independent_and_hides_token(runtime, tmp_path):
    path = connection.write_connection(
        url="http://localhost:1001/mcp", token="do-not-print", app_session_id="one"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run(
        [sys.executable, "-m", "kangengine.adapters.mcp", "sessions"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sessions"][0]["connection_file"] == str(path)
    assert "do-not-print" not in result.stdout + result.stderr


def test_discovery_ignores_dead_but_keeps_unknown_processes(runtime, monkeypatch):
    first = connection.write_connection(
        url="http://localhost:1001/mcp", token="secret", app_session_id="one"
    )
    second = connection.write_connection(
        url="http://localhost:1002/mcp", token="secret", app_session_id="two"
    )
    record = json.loads(second.read_text())
    record["pid"] = 123456
    second.write_text(json.dumps(record))
    real_kill = os.kill

    def dead(pid, signal):
        if pid == 123456:
            raise ProcessLookupError
        return real_kill(pid, signal)

    monkeypatch.setattr(connection.os, "kill", dead)
    assert connection.select_connection_file() == first
    assert connection.list_connections()["sessions"][1]["process_state"] == "dead"
    assert second.exists()

    def unknown(pid, signal):
        raise PermissionError

    monkeypatch.setattr(connection.os, "kill", unknown)
    assert connection.process_state(record) == "unknown"
    with pytest.raises(ValueError, match="Multiple"):
        connection.select_connection_file()


def test_reused_pid_and_missing_identity(runtime, monkeypatch):
    monkeypatch.setattr(connection.os, "kill", lambda *args: None)
    monkeypatch.setattr(connection, "_process_start", lambda pid: "new")
    assert connection.process_state({"pid": 12, "process_start": "old"}) == "dead"
    assert connection.process_state({"pid": 12, "process_start": "new"}) == "alive"
    assert connection.process_state({"pid": 12}) == "alive"
    for pid in (None, 0, -1, True, "12", 2**64):
        assert connection.process_state({"pid": pid}) == "unknown"


def test_connection_helpers_explicit_secret_output(runtime, capsys, monkeypatch):
    from kangengine.adapters.mcp.__main__ import main
    import tomllib

    path = connection.write_connection(
        url="http://127.0.0.1:1001/mcp", token="do-not-leak", app_session_id="one"
    )
    assert (
        main(["--connection-file", str(path), "print-config", "--format", "codex"]) == 0
    )
    output = capsys.readouterr().out
    assert "do-not-leak" not in output
    config = tomllib.loads(output)["mcp_servers"]["kangengine"]
    assert config["url"] == "http://127.0.0.1:1001/mcp"
    assert config["bearer_token_env_var"] == "KANGENGINE_MCP_TOKEN"
    assert main(["print-config"]) == 0
    output = capsys.readouterr().out
    assert "do-not-leak" not in output
    assert json.loads(output)["headers"]["Authorization"] == "Bearer <token>"
    monkeypatch.setenv("KANGENGINE_MCP_TOKEN", "different-environment-credential")
    assert main(["token"]) == 0
    assert capsys.readouterr().out == "do-not-leak\n"

"""Bounded public documentation lookup without engine access or code execution."""

import inspect
from pathlib import Path
import re

from ...app.capture import AutomationError
from ...app.mcp.inspection import _bounded_integer


GUIDE_ROOT = Path(__file__).resolve().parents[4] / "docs/sphinx/guide"
MAX_FILE_BYTES = 256 * 1024


def _root():
    if not GUIDE_ROOT.is_dir():
        raise AutomationError(
            "UNSUPPORTED",
            "Public guide files require a source checkout; use get_api_docs",
        )
    return GUIDE_ROOT.resolve()


def _page(root, path):
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved.suffix not in {".md", ".rst"}:
        raise AutomationError(
            "INVALID_ARGUMENT", "path must name a public guide Markdown/RST page"
        )
    if not resolved.is_file():
        raise AutomationError("NOT_FOUND", "Guide page does not exist")
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise AutomationError("OUTPUT_TOO_LARGE", "Guide page exceeds 256 KiB")
    return resolved.read_text(encoding="utf-8").splitlines()


def search_docs(*, query, max_results=10):
    _bounded_integer(max_results, "max_results", 1, 30)
    if not query.strip() or len(query) > 200:
        raise AutomationError("INVALID_ARGUMENT", "query needs 1..200 characters")
    root = _root()
    query = query.strip().casefold()
    hits = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".md", ".rst"} or not path.is_file():
            continue
        if (
            not path.resolve().is_relative_to(root)
            or path.stat().st_size > MAX_FILE_BYTES
        ):
            continue
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(_page(root, relative), 1):
            if query in line.casefold():
                if len(hits) == max_results:
                    return {"results": hits, "truncated": True}
                hits.append({"path": relative, "line": number, "excerpt": line[:400]})
    return {"results": hits, "truncated": False}


def get_doc(*, path, start_line=1, max_lines=80):
    _bounded_integer(start_line, "start_line", 1, 100000)
    _bounded_integer(max_lines, "max_lines", 1, 120)
    lines = _page(_root(), path)
    selected = lines[start_line - 1 : start_line - 1 + max_lines]
    return {
        "path": path,
        "start_line": start_line,
        "total_lines": len(lines),
        "text": "\n".join(line[:1000] for line in selected),
        "truncated": start_line - 1 + len(selected) < len(lines)
        or any(len(line) > 1000 for line in selected),
    }


def get_api_docs(*, symbol):
    import kangengine

    symbol = symbol.removeprefix("kangengine.").removeprefix("ke.")
    if len(symbol) > 200 or not re.fullmatch(
        r"[A-Za-z][A-Za-z_0-9]*(?:\.[A-Za-z][A-Za-z_0-9]*)*", symbol
    ):
        raise AutomationError("INVALID_ARGUMENT", "symbol must be a public dotted name")
    parts = symbol.split(".")
    if parts[0] not in {
        "App",
        "app",
        "geometry",
        "material",
        "render",
        "scene",
        "visual",
        "asset",
    }:
        raise AutomationError(
            "UNSUPPORTED",
            "Choose App, app, geometry, material, render, scene, visual or asset",
        )
    value = kangengine
    try:
        for part in parts:
            value = inspect.getattr_static(value, part)
    except AttributeError:
        raise AutomationError("NOT_FOUND", "Public symbol does not exist") from None
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = None
    documentation = inspect.getdoc(value) or "No docstring is available."
    return {
        "symbol": f"kangengine.{symbol}",
        "signature": signature,
        "documentation": documentation[:12000],
        "truncated": len(documentation) > 12000,
    }

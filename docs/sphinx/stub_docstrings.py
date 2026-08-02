"""Load public API docstrings from package stub files."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


def _stub_module_name(python_dir: Path, stub_path: Path) -> str:
    relative = stub_path.relative_to(python_dir)
    parts = list(relative.parts)
    if parts[-1] == "__init__.pyi":
        parts.pop()
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts)


def _record_docstring(
    docstrings: dict[str, list[str]], name: str, node: ast.AST
) -> None:
    text = ast.get_docstring(node)
    if text and name not in docstrings:
        docstrings[name] = text.splitlines()


def _index_class(
    docstrings: dict[str, list[str]], module_name: str, node: ast.ClassDef
) -> None:
    class_name = f"{module_name}.{node.name}"
    _record_docstring(docstrings, class_name, node)
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _record_docstring(docstrings, f"{class_name}.{child.name}", child)
        elif isinstance(child, ast.ClassDef):
            _index_class(docstrings, class_name, child)


def load_stub_docstrings(python_dir: Path) -> dict[str, list[str]]:
    """Return qualified public names mapped to non-empty stub docstrings."""
    docstrings: dict[str, list[str]] = {}
    for stub_path in sorted((python_dir / "kangengine").rglob("*.pyi")):
        module_name = _stub_module_name(python_dir, stub_path)
        tree = ast.parse(stub_path.read_text(encoding="utf-8"), stub_path)
        _record_docstring(docstrings, module_name, tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _record_docstring(docstrings, f"{module_name}.{node.name}", node)
            elif isinstance(node, ast.ClassDef):
                _index_class(docstrings, module_name, node)
    return docstrings


def stub_docstring_fingerprint(docstrings: dict[str, list[str]]) -> str:
    """Return a stable digest used to invalidate incremental Sphinx builds."""
    digest = hashlib.sha256()
    for name, lines in sorted(docstrings.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update("\n".join(lines).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

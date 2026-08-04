"""Load public API docstrings from package stub files."""

from __future__ import annotations

import ast
import copy
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


def _without_receiver(arguments: ast.arguments) -> ast.arguments:
    result = copy.deepcopy(arguments)
    positional = result.posonlyargs + result.args
    if not positional or positional[0].arg not in {"self", "cls"}:
        return result
    if result.posonlyargs:
        result.posonlyargs.pop(0)
    else:
        result.args.pop(0)
    return result


def _signature(node: ast.FunctionDef, *, method: bool) -> tuple[str, str | None]:
    arguments = _without_receiver(node.args) if method else node.args
    signature = f"({ast.unparse(arguments)})"
    returns = ast.unparse(node.returns) if node.returns is not None else None
    return signature, returns


def _signature_score(node: ast.FunctionDef) -> tuple[int, int]:
    """Prefer a documented overload, then the declaration with more arguments."""
    count = len(node.args.posonlyargs) + len(node.args.args) + len(node.args.kwonlyargs)
    return (int(bool(ast.get_docstring(node))), count)


def load_stub_signatures(
    python_dir: Path,
) -> dict[str, tuple[str, str | None]]:
    """Return qualified public callables mapped to signatures from stubs.

    Class entries use their ``__init__`` declaration. When a stub has overloads,
    Sphinx receives the most informative declaration while type checkers retain
    every overload from the original ``.pyi`` file.
    """
    signatures: dict[str, tuple[str, str | None]] = {}
    scores: dict[str, tuple[int, int]] = {}

    def record(name: str, node: ast.FunctionDef, *, method: bool) -> None:
        score = _signature_score(node)
        if score >= scores.get(name, (-1, -1)):
            signatures[name] = _signature(node, method=method)
            scores[name] = score

    for stub_path in sorted((python_dir / "kangengine").rglob("*.pyi")):
        module_name = _stub_module_name(python_dir, stub_path)
        tree = ast.parse(stub_path.read_text(encoding="utf-8"), stub_path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                record(f"{module_name}.{node.name}", node, method=False)
            elif isinstance(node, ast.ClassDef):
                class_name = f"{module_name}.{node.name}"
                for child in node.body:
                    if not isinstance(child, ast.FunctionDef):
                        continue
                    decorators = {
                        ast.unparse(decorator) for decorator in child.decorator_list
                    }
                    if "property" in decorators or any(
                        decorator.endswith(".setter")
                        for decorator in decorators
                    ):
                        continue
                    if child.name == "__init__":
                        record(class_name, child, method=True)
                    elif not child.name.startswith("_"):
                        record(f"{class_name}.{child.name}", child, method=True)
    return signatures


def stub_docstring_fingerprint(docstrings: dict[str, list[str]]) -> str:
    """Return a stable digest used to invalidate incremental Sphinx builds."""
    digest = hashlib.sha256()
    for name, lines in sorted(docstrings.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update("\n".join(lines).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stub_signature_fingerprint(
    signatures: dict[str, tuple[str, str | None]],
) -> str:
    """Return a stable digest used to invalidate signature documentation."""
    digest = hashlib.sha256()
    for name, signature in sorted(signatures.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(signature).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

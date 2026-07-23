"""Check that public package exports are represented in package stubs."""

from __future__ import annotations

import ast
from pathlib import Path

import kangengine as ke


def _declared_names(stub_path: Path) -> set[str]:
    tree = ast.parse(stub_path.read_text(), filename=str(stub_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return names


def main() -> None:
    package_root = Path(ke.__file__).resolve().parent
    for package_name in ("physics", "render"):
        module = getattr(ke, package_name)
        stub_path = package_root / package_name / "__init__.pyi"
        if not stub_path.is_file():
            raise AssertionError(f"missing public package stub: {stub_path}")
        missing = set(module.__all__) - _declared_names(stub_path)
        if missing:
            raise AssertionError(
                f"{package_name} stub is missing public exports: {sorted(missing)}"
            )
    print("PASS: public package stubs cover runtime exports")


if __name__ == "__main__":
    main()

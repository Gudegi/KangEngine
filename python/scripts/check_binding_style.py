"""Reject Python-facing pybind names that violate KangEngine API style."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BINDINGS = ROOT / "bindings"
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
UPPER_SNAKE_CASE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
PASCAL_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
EXTERNAL_ENUM_CONTRACTS = {"MimicControlMode", "MimicObjType"}


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main() -> int:
    errors: list[str] = []
    for path in sorted(BINDINGS.glob("*.cpp")):
        text = path.read_text()
        relative = path.relative_to(ROOT)

        method_pattern = re.compile(
            r'\.def(?:_static|_property|_property_readonly)?\("([A-Za-z][A-Za-z0-9_]*)"'
        )
        for match in method_pattern.finditer(text):
            name = match.group(1)
            if not SNAKE_CASE.fullmatch(name):
                errors.append(
                    f"{relative}:{line_number(text, match.start())}: "
                    f"Python method/property must be snake_case: {name}"
                )

    package_root = ROOT / "python" / "kangengine"
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(ROOT)
        tree = ast.parse(path.read_text(), filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and not SNAKE_CASE.fullmatch(
                    node.name
                ):
                    errors.append(
                        f"{relative}:{node.lineno}: Python function/method must "
                        f"be snake_case: {node.name}"
                    )
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_") and not PASCAL_CASE.fullmatch(
                    node.name
                ):
                    errors.append(
                        f"{relative}:{node.lineno}: Python class must be "
                        f"PascalCase: {node.name}"
                    )
                is_enum = any(
                    isinstance(base, ast.Name) and base.id == "Enum"
                    for base in node.bases
                )
                if not is_enum or node.name in EXTERNAL_ENUM_CONTRACTS:
                    continue
                for child in node.body:
                    if isinstance(child, (ast.Assign, ast.AnnAssign)):
                        targets = (
                            child.targets
                            if isinstance(child, ast.Assign)
                            else [child.target]
                        )
                        for target in targets:
                            if isinstance(target, ast.Name) and not UPPER_SNAKE_CASE.fullmatch(
                                target.id
                            ):
                                errors.append(
                                    f"{relative}:{child.lineno}: Python enum value "
                                    f"must be UPPER_SNAKE_CASE: {target.id}"
                                )

        member_pattern = re.compile(r'\.def_(?:readwrite|readonly)\("([A-Za-z][A-Za-z0-9_]*)"')
        for match in member_pattern.finditer(text):
            name = match.group(1)
            if not SNAKE_CASE.fullmatch(name):
                errors.append(
                    f"{relative}:{line_number(text, match.start())}: "
                    f"Python field must be snake_case: {name}"
                )

        for match in re.finditer(r'py::arg(?:_v)?\("([A-Za-z][A-Za-z0-9_]*)"', text):
            name = match.group(1)
            if not SNAKE_CASE.fullmatch(name):
                errors.append(
                    f"{relative}:{line_number(text, match.start())}: "
                    f"Python argument must be snake_case: {name}"
                )

        for match in re.finditer(r'\.value\("([A-Za-z][A-Za-z0-9_]*)"', text):
            name = match.group(1)
            if not UPPER_SNAKE_CASE.fullmatch(name):
                errors.append(
                    f"{relative}:{line_number(text, match.start())}: "
                    f"Python enum value must be UPPER_SNAKE_CASE: {name}"
                )

        type_pattern = re.compile(
            r"py::(?:class_|enum_)<.*?>\s*\(.*?,\s*\"([A-Za-z][A-Za-z0-9_]*)\"",
            re.DOTALL,
        )
        for match in type_pattern.finditer(text):
            name = match.group(1)
            if not PASCAL_CASE.fullmatch(name):
                errors.append(
                    f"{relative}:{line_number(text, match.start())}: "
                    f"Python type must be PascalCase: {name}"
                )

    if errors:
        print("Python binding style violations:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("PASS: Python binding names follow the public API style")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

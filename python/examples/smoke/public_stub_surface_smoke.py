"""Check that public package exports are represented in package stubs."""

from __future__ import annotations

import ast
import inspect
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


def _stub_function_parameters(stub_path: Path) -> dict[str, tuple[str, ...]]:
    tree = ast.parse(stub_path.read_text(), filename=str(stub_path))
    return {
        node.name: tuple(argument.arg for argument in node.args.args)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _missing_public_docstrings(stub_path: Path, public_names: set[str]) -> list[str]:
    tree = ast.parse(stub_path.read_text(), filename=str(stub_path))
    missing: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        if node.name not in public_names:
            continue
        if not ast.get_docstring(node):
            missing.append(node.name)
        if not isinstance(node, ast.ClassDef):
            continue
        method_docs: dict[str, bool] = {}
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or child.name.startswith("_"):
                continue
            method_docs[child.name] = method_docs.get(child.name, False) or bool(
                ast.get_docstring(child)
            )
        missing.extend(
            f"{node.name}.{name}"
            for name, documented in method_docs.items()
            if not documented
        )
    return missing


def _native_function_parameters(function) -> tuple[str, ...]:
    signature = function.__doc__.splitlines()[0]
    parsed = ast.parse(f"def {signature}: ...")
    definition = parsed.body[0]
    return tuple(argument.arg for argument in definition.args.args)


def main() -> None:
    package_root = Path(ke.__file__).resolve().parent
    for package_name in ("animation", "physics", "render"):
        module = getattr(ke, package_name)
        package_stub = package_root / package_name / "__init__.pyi"
        module_stub = package_root / f"{package_name}.pyi"
        stub_path = package_stub if package_stub.is_file() else module_stub
        if not stub_path.is_file():
            raise AssertionError(f"missing public package stub: {stub_path}")
        missing = set(module.__all__) - _declared_names(stub_path)
        if missing:
            raise AssertionError(
                f"{package_name} stub is missing public exports: {sorted(missing)}"
            )
        if package_name == "animation":
            missing_docs = _missing_public_docstrings(stub_path, set(module.__all__))
            if missing_docs:
                raise AssertionError(
                    "animation stub is missing public docstrings: "
                    f"{sorted(missing_docs)}"
                )
            stub_parameters = _stub_function_parameters(stub_path)
            for name in module.__all__:
                function = getattr(module, name)
                if not inspect.isbuiltin(function):
                    continue
                native_parameters = _native_function_parameters(function)
                if stub_parameters.get(name) != native_parameters:
                    raise AssertionError(
                        f"animation.{name} stub parameters "
                        f"{stub_parameters.get(name)} do not match native "
                        f"parameters {native_parameters}"
                    )
    print("PASS: public package stubs cover runtime exports and checked signatures")


if __name__ == "__main__":
    main()

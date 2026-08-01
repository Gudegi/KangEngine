from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

from docutils import nodes

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"

sys.path.insert(0, str(PYTHON_DIR))

# The extension exposes imgui and keys as module-valued public attributes
# rather than importable Python packages. Register documentation-only aliases
# so autodoc indexes their members under the public paths.
import kangengine as _ke_docs

sys.modules.setdefault("kangengine.imgui", _ke_docs.imgui)
sys.modules.setdefault("kangengine.keys", _ke_docs.keys)

project = "KangEngine"
author = "KangEngine contributors"
copyright = "2026, KangEngine contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/app.md",
    "api/sensor.md",
]

html_theme = os.environ.get("KANGENGINE_SPHINX_THEME", "furo")
html_title = "KangEngine Docs"
html_static_path = ["_static"]
html_css_files = ["kangengine.css"]
html_js_files = ["kangengine-toc.js"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
python_maximum_signature_line_length = 88
python_trailing_comma_in_multi_line_signatures = True
python_use_unqualified_type_names = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

# The Python API reference expects the pybind11 extension to exist. Build it
# first with `make build_python` or `make build_usd_python`.
nitpicky = False

_PUBLIC_SIGNATURE_REPLACEMENTS = (
    ("kangengine._core._ke.animation.", "kangengine.animation."),
    ("kangengine._kangengine.animation.", "kangengine.animation."),
    ("kangengine._core._ke.scene.", "kangengine.scene."),
    ("kangengine._kangengine.scene.", "kangengine.scene."),
    ("typing.SupportsFloat", "float"),
    ("typing.SupportsInt", "int"),
    ("SupportsFloat", "float"),
    ("SupportsInt", "int"),
    ("kangengine._core._ke.asset.", "kangengine.asset."),
    ("kangengine._kangengine.asset.", "kangengine.asset."),
    ("kangengine._core._ke.physics.", "kangengine.physics."),
    ("kangengine._kangengine.physics.", "kangengine.physics."),
    ("kangengine._core._ke.", "kangengine."),
    ("kangengine._kangengine.", "kangengine."),
)


def _public_signature(text: str | None) -> str | None:
    if text is None:
        return None
    for private, public in _PUBLIC_SIGNATURE_REPLACEMENTS:
        text = text.replace(private, public)
    return text


def process_signature(app, what, name, obj, options, signature, return_annotation):
    signature = _public_signature(signature)
    return_annotation = _public_signature(return_annotation)

    if name.endswith(".__init__") and signature:
        first_comma = signature.find(",")
        if signature.startswith("(self:") and first_comma >= 0:
            signature = "(" + signature[first_comma + 1 :].lstrip()
        return_annotation = None

    return signature, return_annotation


def process_docstring(app, what, name, obj, options, lines):
    constant_modules = {
        "kangengine.imgui": "Window flags",
        "kangengine.keys": "Key constants",
    }
    title = constant_modules.get(name)
    if what != "module" or title is None:
        return

    constants = [
        (member_name, getattr(obj, member_name))
        for member_name in dir(obj)
        if not member_name.startswith("_")
        and isinstance(getattr(obj, member_name), int)
    ]
    if not constants:
        return

    lines.extend(
        [
            "",
            title,
            "~" * len(title),
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 70 30",
            "",
            "   * - Name",
            "     - Value",
        ]
    )
    for member_name, value in constants:
        public_module = "ke.imgui" if name.endswith(".imgui") else "ke.keys"
        lines.extend(
            [
                f"   * - ``{public_module}.{member_name}``",
                f"     - ``{value}``",
            ]
        )


def process_bases(app, name, obj, options, bases):
    bases[:] = [
        base
        for base in bases
        if base is not object and getattr(base, "__module__", "") != "pybind11_builtins"
    ]


def _documented_object(signature):
    module_name = signature.get("module")
    fullname = signature.get("fullname")
    if not module_name or not fullname:
        return None

    try:
        obj = importlib.import_module(module_name)
        for part in fullname.split("."):
            obj = getattr(obj, part)
        return obj
    except (AttributeError, ImportError):
        return None


def process_object_description(app, domain, objtype, contentnode):
    if domain != "py" or objtype not in {"class", "function"}:
        return
    for child in tuple(contentnode.children):
        if isinstance(child, nodes.paragraph) and child.astext() == "Bases:":
            contentnode.remove(child)

    description = contentnode.parent
    if not description.children:
        return
    signature = description.children[0]
    obj = _documented_object(signature)
    if obj is None:
        return

    is_native = inspect.isbuiltin(obj) or type(obj).__module__ == "pybind11_builtins"
    kind = "native" if is_native else "python"
    signature += nodes.inline(
        "",
        f"[{kind}]",
        classes=["api-origin", f"api-origin-{kind}"],
    )


def setup(app):
    app.connect("autodoc-process-signature", process_signature)
    app.connect("autodoc-process-bases", process_bases)
    app.connect("object-description-transform", process_object_description)
    app.connect("autodoc-process-docstring", process_docstring)

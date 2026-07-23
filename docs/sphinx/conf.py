from __future__ import annotations

import os
import sys
from pathlib import Path


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
    return _public_signature(signature), _public_signature(return_annotation)


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


def setup(app):
    app.connect("autodoc-process-signature", process_signature)
    app.connect("autodoc-process-docstring", process_docstring)

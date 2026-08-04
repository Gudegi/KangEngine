"""Validate generated Sphinx API signatures and docstring sources."""

from __future__ import annotations

import re
import sys
from html import unescape
from pathlib import Path

FORBIDDEN_TEXT = {
    "verbose pybind ArrayLike annotation": "Annotated[numpy.typing.ArrayLike",
    "typing protocol scalar": "typing.SupportsFloat",
    "typing protocol integer": "typing.SupportsInt",
    "pybind implementation base": "pybind11_object",
    "empty base-class row": "<p>Bases:</p>",
}

REQUIRED_TEXT = {
    "animation.html": {
        "stub docstring": "Shapes use V=vertices, B=skin bones, and S=skeleton nodes.",
    },
    "physics.html": {
        "native fallback": "PhysX world configuration including timestep, up axis, and reporting.",
    },
    "rendering.html": {
        "native fallback": "Factory for backend graphics resources such as textures and buffers.",
    },
    "visual.html": {
        "simulation visual section": "Simulation visual sync",
        "simulation visual ownership": "simulation state remains owned by",
    },
}

REQUIRED_HTML = {
    "index.html": {
        "GitHub repository link": 'href="https://github.com/Gudegi/KangEngine"',
        "GitHub accessible label": 'aria-label="KangEngine on GitHub"',
        "View this page button": 'title="View this page"',
        "View this page icon": 'href="#svg-eye"',
    },
}


def main() -> int:
    api_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_build/html/api")
    failures: list[str] = []
    for path in sorted(api_dir.glob("*.html")):
        html = path.read_text(encoding="utf-8")
        text = unescape(re.sub(r"<[^>]+>", "", html))
        for label, forbidden in FORBIDDEN_TEXT.items():
            if forbidden in html or forbidden in text:
                failures.append(f"{path}: {label}")
        for label, required in REQUIRED_TEXT.get(path.name, {}).items():
            if required not in text:
                failures.append(f"{path}: missing {label}")
        for label, required in REQUIRED_HTML.get(path.name, {}).items():
            if required not in html:
                failures.append(f"{path}: missing {label}")

    if failures:
        print("Public API output validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("PASS: generated public API output passed signature and docstring checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

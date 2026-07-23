# Verify Installation

First verify that Python can import the package and report its optional USD
capability.

```bash
python/.venv/bin/python -c \
  "import kangengine as ke; print(ke.__file__); print('USD:', ke.scene.has_usd_support())"
```

A normal wheel or `make build_python` build should print `USD: False`.

## Verify the public API

```bash
make validate_python_api
```

This builds the extension and checks the canonical package surface and type
information.

## Open a window

```bash
python ./python/examples/render_prim_scene.py
```

Expected result: a checkerboard ground, an orange box, and a blue sphere.

If the import succeeds but the window does not open, verify the platform OpenGL
requirements before debugging Python packaging.

Next: [Hello App](HELLO_APP.md).

# Installation

Install KangEngine from PyPI in a Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install kangengine
```

If you use uv, run `uv pip install kangengine` in your active environment.

## Requirements

The current release provides these wheels:

| Platform | Requirements | Simulation |
| --- | --- | --- |
| macOS | macOS 26+, Apple Silicon, CPython 3.12 | PhysX CPU |
| Linux | x86-64, glibc 2.39+ (Ubuntu 24.04 tested), CPython 3.12 | PhysX CPU/GPU |

Linux also requires an NVIDIA driver version 580+ and a compatible GPU
(tested on RTX 4090). pip installs the CUDA runtime and PhysX GPU packages
automatically; a separate CUDA Toolkit is not required for wheel installation.

Both wheels include OpenUSD. Opening a viewer requires a graphical session
with OpenGL 4.1 support.

## Verify installation

```bash
python -c "import kangengine as ke; print(ke.__file__); print('USD:', ke.scene.has_usd_support())"
```

Expect `USD: True`. Continue with [Hello App](HELLO_APP.md) to create a visible
scene, or [Verify Installation](VERIFY_INSTALLATION.md) if installation fails.

## Development installation

For editable installs, C++ builds, or platforms without a matching wheel, use
[Build from Source](../advanced/BUILD_FROM_SOURCE.md).

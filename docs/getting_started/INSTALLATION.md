# Installation

KangEngine currently ships as platform-specific wheels built from a source
checkout. Use Python 3.12 and build the wheel on the same OS and architecture
family where it will run.

## Requirements

- macOS on Apple Silicon, or Ubuntu 24.04
- Python 3.12
- CMake, a C++17 compiler, and vcpkg
- PhysX under `$HOME/Physics/PhysX`

Linux/NVIDIA GPU simulation additionally requires the supported PhysX 5.8 GPU
build and CUDA toolkit. Start with the CPU build unless GPU simulation is the
reason you are installing KangEngine.

## Create the development environment

From the repository root:

```bash
uv venv python/.venv --python 3.12
source python/.venv/bin/activate
uv sync --project python
```

## Build the Python extension

```bash
make build_python
```

The normal Python build intentionally disables OpenUSD. To build an optional
USD-enabled development extension instead:

```bash
make build_usd_python
```

## Install for development

```bash
uv pip install -e ./python
```

## Install a Built Wheel

If you received a wheel built for your operating system, architecture, and
Python 3.12 ABI, install it into an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install /path/to/kangengine-0.1.0-cp312-cp312-platform.whl
python -c "import kangengine as ke; print(ke.__file__)"
```

KangEngine wheels are platform-specific. The Linux CUDA wheel additionally
requires a compatible NVIDIA driver and the GPU runtime described in
[PhysX GPU Simulation](../advanced/PHYSX_GPU.md). Distributed wheels do not
include OpenUSD support.

Continue with [Verify Installation](VERIFY_INSTALLATION.md).

For platform-specific PhysX setup, source builds, CUDA 13 notes, and wheel
production, see [Build from Source](../advanced/BUILD_FROM_SOURCE.md).

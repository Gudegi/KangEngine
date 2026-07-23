# Build Guide

This guide contains the detailed build setup for KangEngine. The root README keeps only the short path; use this document when setting up a new machine or enabling optional components.

## Requirements

- CMake
- Ninja or a compatible build tool
- A C++17 compiler
- vcpkg
- PhysX under `$HOME/Physics/PhysX` (5.1 CPU compatibility or 5.8 GPU)
- Python 3.12 for Python bindings

## vcpkg

KangEngine uses vcpkg manifest mode for most third-party C++ dependencies.

1. Clone and bootstrap vcpkg.

    ```bash
    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg && ./bootstrap-vcpkg.sh
    ```

2. Export `VCPKG_ROOT` and add vcpkg to `PATH`.

    ```bash
    export VCPKG_ROOT=/path/to/vcpkg
    export PATH=$VCPKG_ROOT:$PATH
    ```

## Linux

Tested with Ubuntu 24.04.

1. Install system packages.

    ```bash
    sudo apt install clang ninja-build unzip libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config autoconf autoconf-archive automake libtool
    ```

2. Choose and build one PhysX configuration.

<details>
<summary>CPU build — PhysX 5.1.2</summary>

Download NVIDIA Omniverse PhysX under `$HOME/Physics/PhysX`.

```bash
mkdir -p ~/Physics
cd ~/Physics
wget https://github.com/NVIDIA-Omniverse/PhysX/archive/refs/tags/104.1-physx-5.1.2.zip
unzip 104.1-physx-5.1.2.zip
mv PhysX-104.1-physx-5.1.2 PhysX
```

Build it with clang.

```bash
cd ~/Physics/PhysX/physx
./buildtools/packman/packman update -y
./generate_projects.sh

cd compiler/linux-release
cmake . \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-Wno-error=unsafe-buffer-usage -Wno-unsafe-buffer-usage -Wno-error=switch-default -Wno-switch-default -Wno-error=invalid-offsetof -Wno-invalid-offsetof -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable"
cmake --build . --config release
```

KangEngine uses the libraries in
`~/Physics/PhysX/physx/bin/linux.clang/release`. PhysX snippet executables may
fail to link against their bundled OpenGL package; KangEngine does not require
those executables.

</details>

<details>
<summary>GPU build — PhysX 5.8</summary>

Clone the NVIDIA Omniverse PhysX
[`110.0-omni-and-physx-5.8.0`](https://github.com/NVIDIA-Omniverse/PhysX/tree/110.0-omni-and-physx-5.8.0)
tag under `$HOME/Physics/PhysX`.

```bash
mkdir -p ~/Physics
cd ~/Physics
git clone \
  --branch 110.0-omni-and-physx-5.8.0 \
  --depth 1 \
  https://github.com/NVIDIA-Omniverse/PhysX.git \
  PhysX
```

Follow PhysX's official
[`README_LINUX.md`](https://github.com/NVIDIA-Omniverse/PhysX/blob/110.0-omni-and-physx-5.8.0/physx/documentation/platformreadme/linux/README_LINUX.md)
to install its Linux prerequisites and generate the build. Enable the PhysX GPU
projects and produce a release build under `linux.x86_64`.

KangEngine's CUDA targets select that output through:

```bash
PHYSX_CUDA_BIN_PLATFORM=linux.x86_64
```

The expected GPU shared library is:

```text
$HOME/Physics/PhysX/physx/bin/linux.x86_64/release/libPhysXGpu_64.so
```

PhysX 5.8 supports CUDA 12.8 directly. Building it with CUDA 13 requires the
compatibility patch below; the `sm_89` verification applies to RTX 4090 and
other Ada targets.

### CUDA 13 / RTX 4090 compatibility

<details>
<summary>PhysX 5.8 architecture and CUDA Driver API patches</summary>

#### TL;DR

For PhysX 5.8, CUDA 13, and RTX 4090:

```bash
export PHYSX_ROOT=$HOME/Physics/PhysX/physx
export PHYSX_BUILD=$PHYSX_ROOT/compiler/linux-clang-release-5.8
export PHYSX_CUDA_SOURCE=$PHYSX_ROOT/source/cudamanager/src/CudaContextManager.cpp

python3 - "$PHYSX_CUDA_SOURCE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old = "status = cuCtxCreate(&mCtx, (unsigned int)flags, mDevHandle);"
new = """#if CUDA_VERSION >= 13000
CUctxCreateParams ctxCreateParams = {};
status = cuCtxCreate(&mCtx, &ctxCreateParams, (unsigned int)flags, mDevHandle);
#else
status = cuCtxCreate(&mCtx, (unsigned int)flags, mDevHandle);
#endif"""
if new not in source:
    if old not in source:
        raise SystemExit("cuCtxCreate call not found; inspect the PhysX source")
    path.write_text(source.replace(old, new, 1))
PY

cmake -S $PHYSX_ROOT/compiler/public \
  -B $PHYSX_BUILD \
  -DPHYSX_ROOT_DIR=$PHYSX_ROOT \
  -DTARGET_BUILD_PLATFORM=linux \
  -DCMAKE_BUILD_TYPE=release \
  -DPX_OUTPUT_LIB_DIR=$PHYSX_ROOT \
  -DPX_OUTPUT_BIN_DIR=$PHYSX_ROOT \
  -DPX_GENERATE_STATIC_LIBRARIES=ON \
  -DPX_GENERATE_GPU_PROJECTS=ON \
  -DPX_GENERATE_GPU_REDUCED_ARCHITECTURES=ON

rg 'ARCH_CODE_LIST' $PHYSX_BUILD/CMakeCache.txt
cmake --build $PHYSX_BUILD --parallel
/usr/local/cuda/bin/cuobjdump --list-elf \
  $PHYSX_ROOT/bin/linux.x86_64/release/libPhysXGpu_64.so \
  | rg 'sm_89'
```

<details>
<summary>Why these flags and patches are needed</summary>

These notes document the Linux build that produced a PhysX 5.8.0 GPU library
with native `sm_89` cubins for RTX 4090. Use this path when the older PhysX
5.1 GPU binary reports PhysX internal CUDA kernel launch failures on Ada GPUs.

The examples below assume the PhysX checkout lives at:

```bash
$HOME/Physics/PhysX
```

The build directory name is not special. Use any clean directory outside older
PhysX build trees; this document uses:

```bash
$HOME/Physics/PhysX/physx/compiler/linux-clang-release-5.8
```

#### CUDA 13 architecture fix

CUDA 13.0 `nvcc` no longer accepts `compute_70`. PhysX 5.8.0's default GPU
architecture list includes it unless reduced GPU architectures are enabled.
Configure with `PX_GENERATE_GPU_REDUCED_ARCHITECTURES=ON` so the generated
`ARCH_CODE_LIST` starts at `compute_80` and includes `compute_89`.

This is the minimal direct CMake invocation used for KangEngine:

```bash
export PHYSX_ROOT=$HOME/Physics/PhysX/physx
export PHYSX_BUILD=$PHYSX_ROOT/compiler/linux-clang-release-5.8

cmake -S $PHYSX_ROOT/compiler/public \
  -B $PHYSX_BUILD \
  -DPHYSX_ROOT_DIR=$PHYSX_ROOT \
  -DTARGET_BUILD_PLATFORM=linux \
  -DCMAKE_BUILD_TYPE=release \
  -DPX_OUTPUT_LIB_DIR=$PHYSX_ROOT \
  -DPX_OUTPUT_BIN_DIR=$PHYSX_ROOT \
  -DPX_GENERATE_STATIC_LIBRARIES=ON \
  -DPX_GENERATE_GPU_PROJECTS=ON \
  -DPX_GENERATE_GPU_REDUCED_ARCHITECTURES=ON
```

`PHYSX_BUILD` can point to any clean build directory. Keep the remaining options
explicit: PhysX's public CMake entry point requires the root/output paths, while
KangEngine needs static PhysX libraries, GPU projects, and the reduced CUDA
architecture list for CUDA 13.

After configure, confirm `compute_70` is gone:

```bash
rg 'ARCH_CODE_LIST' \
  $PHYSX_BUILD/CMakeCache.txt
```

Expected `ARCH_CODE_LIST` includes:

```text
compute_80, compute_86, compute_89, compute_90, compute_100, compute_120
```

#### CUDA 13 `cuCtxCreate` patch

PhysX 5.8.0 calls the older 3-argument CUDA Driver API form:

```cpp
cuCtxCreate(&mCtx, (unsigned int)flags, mDevHandle);
```

With CUDA 13 headers, `cuCtxCreate` maps to `cuCtxCreate_v4`, which expects a
`CUctxCreateParams*` argument. Patch
`$PHYSX_ROOT/source/cudamanager/src/CudaContextManager.cpp`
near the CUDA context creation call:

```cpp
#if CUDA_VERSION >= 13000
CUctxCreateParams ctxCreateParams = {};
status = cuCtxCreate(&mCtx, &ctxCreateParams, (unsigned int)flags, mDevHandle);
#else
status = cuCtxCreate(&mCtx, (unsigned int)flags, mDevHandle);
#endif
```

Then build:

```bash
cmake --build \
  $PHYSX_BUILD \
  --parallel
```

The build should finish with:

```text
[100%] Built target PhysXVehicle2
```

#### Verify native Ada GPU kernels

Check that the rebuilt GPU library contains `sm_89` cubins:

```bash
/usr/local/cuda/bin/cuobjdump --list-elf \
  $PHYSX_ROOT/bin/linux.x86_64/release/libPhysXGpu_64.so \
  | rg 'sm_89'
```

You should see entries such as:

```text
broadphase.sm_89.cubin
MemCopyBalanced.sm_89.cubin
solver.sm_89.cubin
solverTGS.sm_89.cubin
integrationTGS.sm_89.cubin
```

Use this PhysX binary directory when linking KangEngine against the 5.8.0
checkout:

```bash
$PHYSX_ROOT/bin/linux.x86_64/release
```

</details>

</details>

</details>

Do not run PhysX configure/build commands with `sudo`. If an earlier build
created root-owned files, repair their ownership:

```bash
sudo chown -R "$USER:$USER" ~/Physics/PhysX/physx
```

3. Build KangEngine for the selected PhysX configuration.

<details>
<summary>CPU KangEngine build</summary>

```bash
CC=clang CXX=clang++ cmake --preset=vcpkg
cmake --build build/release
```

</details>

<details>
<summary>GPU KangEngine build</summary>

KangEngine links `PhysXGpu_64`; its shared library must be discoverable at
runtime.

```bash
make build_all
make build_cuda
make build_python_cuda
```

Run `make validate_physx_gpu` for the process-isolated Python GPU regression
suite or `make validate_physx_gpu_cpp` for the native smoke test. See
[`PHYSX_GPU.md`](PHYSX_GPU.md) for the runtime contract.

</details>

4. Run KangEngine.

    ```bash
    make run2
    ```

## macOS

Tested with Apple Silicon.

1. Clone o3de PhysX under `$HOME/Physics/PhysX`.

    ```bash
    mkdir -p ~/Physics
    cd ~/Physics
    git clone -b 104.1 https://github.com/o3de/PhysX.git
    ```

2. Install build tools.

    ```bash
    brew install coreutils ninja autoconf automake autoconf-archive
    ```

3. Build PhysX.

    ```bash
    cd ~/Physics/PhysX/physx
    ./buildtools/packman/packman update -y
    ./generate_projects.sh

    # The O3DE PhysX build system uses the 'mac.x86_64' directory name for all macOS builds, including Apple Silicon.
    cd compiler/mac.x86_64
    cmake --build . --config release
    ```

4. Configure KangEngine.

    ```bash
    cmake --preset=vcpkg
    ```

5. Build KangEngine.

    ```bash
    cmake --build build/release
    ```

## Build Targets

Common make targets:

```bash
make build
make build_debug
make build_python
make build_all
make build_cuda
make build_python_cuda
make build_usd
make build_usd_python
make wheel
make wheel_cuda
make validate_wheel
make validate_wheel_cuda
make validate_physx_gpu
make validate_physx_gpu_cpp
make run2
```

The executable target is selected in `CMakeLists.txt` by changing the active `MAIN_FILE` entry near the example list.

## OpenUSD Optional

OpenUSD is only needed when configuring KangEngine with `-DUSE_USD=ON`.

1. Clone OpenUSD.

    ```bash
    cd ~
    git clone https://github.com/PixarAnimationStudios/OpenUSD.git
    ```

2. Build OpenUSD into `~/usd_build`.

    ```bash
    mkdir -p ~/usd_build
    python3 ~/OpenUSD/build_scripts/build_usd.py ~/usd_build
    ```

3. If you build OpenUSD somewhere else, pass
   `-DUSD_DIR=/path/to/usd_build` to your CMake configure command. `USD_DIR`
   is the OpenUSD installation prefix containing `include/` and `lib/`.

## Python Bindings Optional

KangEngine exposes a Python module, `kangengine`, via pybind11. The extension is
built by CMake and must match the consumer's CPython ABI, Python minor version,
platform, and architecture.

1. Create a virtual environment with Python 3.12 using `uv`.

    ```bash
    uv venv python/.venv --python 3.12
    source python/.venv/bin/activate
    ```

2. Build the extension from the repo root.

    ```bash
    make build_python
    ```

    Or with USD support:

    ```bash
    make build_usd_python
    ```

3. Install the Python package in editable mode.

    ```bash
    uv pip install -e ./python
    ```

4. Run an example.

    ```bash
    python ./python/examples/view_bvh_character.py
    ```

## Python Wheels

KangEngine builds separate native wheels on each target platform. Wheel builds
use `python/.venv/bin/python` by default, so activating the development virtual
environment is optional. Create and populate that environment as described in
the previous section before building a wheel.

The distributed wheels intentionally disable OpenUSD. This keeps the native
extension independent of OpenUSD, TBB, and USD plugin resources. USD-enabled
development builds remain available through `make build_usd_python`, but a
USD-enabled distribution wheel is not currently produced.

Build and preserve a wheel under `python/dist`:

```bash
# macOS, CPU PhysX
make wheel

# Linux, CUDA and the configured GPU PhysX build
make wheel_cuda
```

The filename records the active CPython ABI and platform, for example:

```text
python/dist/kangengine-0.1.0-cp312-cp312-macosx_26_0_arm64.whl
```

Wheel creation uses a temporary staging directory, so stale files under a
previous setuptools build directory cannot enter the package. The
`kangengine/assets/external` directory is excluded; the remaining runtime
assets, Python modules, type information, and `_kangengine.so` are included.

To build, install, and test a temporary wheel without changing the development
environment, run:

```bash
make validate_wheel
make validate_wheel_cuda  # Linux CUDA host only
```

Validation checks the native platform tag, no-USD build policy, package
contents, public API, and type-stub surface. It installs KangEngine into a
temporary target directory, reuses the development environment's Python
dependencies, and removes the temporary wheel and installation afterward.

To test the preserved wheel as a consumer, create a separate environment with
the matching Python version and install the wheel there:

```bash
python3.12 -m venv /tmp/kangengine-wheel-venv
source /tmp/kangengine-wheel-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install python/dist/kangengine-0.1.0-cp312-cp312-macosx_26_0_arm64.whl
python -c "import kangengine as ke; assert not ke.scene.has_usd_support()"
```

Use the corresponding filename emitted by `make wheel_cuda` on Linux.

The macOS wheel contains the statically linked PhysX CPU libraries. The Linux
CUDA wheel still requires a compatible NVIDIA driver, CUDA runtime policy, and
GPU PhysX environment for full simulation validation. Build and test each
wheel on the same operating-system and architecture family on which it will be
distributed.

## Build the Documentation

Build the Python extension first so the API reference can import the pybind11
module:

```bash
make build_python
make docs
```

To include APIs that exist only in a USD-enabled build:

```bash
make build_usd_python
make docs
```

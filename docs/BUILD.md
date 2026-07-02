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

2. Download NVIDIA Omniverse PhysX under `$HOME/Physics/PhysX`.

    ```bash
    mkdir -p ~/Physics
    cd ~/Physics
    wget https://github.com/NVIDIA-Omniverse/PhysX/archive/refs/tags/104.1-physx-5.1.2.zip
    unzip 104.1-physx-5.1.2.zip
    mv PhysX-104.1-physx-5.1.2 PhysX
    ```

3. Build PhysX with clang.

    ```bash
    cd ~/Physics/PhysX/physx
    ./buildtools/packman/packman update -y
    ./generate_projects.sh

    cd compiler/linux-release
    cmake . \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CXX_FLAGS="-Wno-error=unsafe-buffer-usage -Wno-unsafe-buffer-usage -Wno-error=switch-default -Wno-switch-default -Wno-error=invalid-offsetof -Wno-invalid-offsetof -Wno-error=unused-but-set-variable -Wno-unused-but-set-variable"
    cmake --build . --config release # (debug|checked|profile|release)
    ```

    Do not run the PhysX configure/build commands with `sudo`. If files were created as root, fix ownership first.

    ```bash
    sudo chown -R "$USER:$USER" ~/Physics/PhysX/physx
    ```

    The PhysX snippet executables may fail to link against the bundled OpenGL package. KangEngine only needs the PhysX libraries in `~/Physics/PhysX/physx/bin/linux.clang/release`.

4. Configure KangEngine with clang.

    ```bash
    CC=clang CXX=clang++ cmake --preset=vcpkg
    ```

5. Build KangEngine.

    ```bash
    cmake --build build/release
    ```

6. Run KangEngine.

    ```bash
    make run2
    ```

### PhysX GPU On Linux

If you build PhysX with GPU support, KangEngine links `PhysXGpu_64` on Linux.
The shared library must be discoverable at runtime. The normal CPU-compatible
build defaults to `linux.clang`; CUDA make targets select the PhysX 5.8
`linux.x86_64` output through `PHYSX_CUDA_BIN_PLATFORM`.

```bash
make build_all
make build_cuda
make build_python_cuda
```

PhysX GPU support is still experimental in KangEngine, especially when used from Python together with Torch CUDA.

For the reproducible Linux validation procedure, see
[`PHYSX_GPU_VALIDATION.md`](PHYSX_GPU_VALIDATION.md).

<details>
<summary>[EXPERIMENTAL] PhysX 5.8.0 GPU build notes for CUDA 13 / RTX 4090</summary>

These notes document the Linux build that produced a PhysX 5.8.0 GPU library
with native `sm_89` cubins for RTX 4090. Use this path when the older PhysX
5.1 GPU binary reports PhysX internal CUDA kernel launch failures on Ada GPUs.

The tested checkout lives at:

```bash
/home/asaid/Physics/PhysX
```

The build directory name is not special. Use any clean directory outside older
PhysX build trees; this document uses:

```bash
/home/asaid/Physics/PhysX/physx/compiler/linux-clang-release-5.8
```

### CUDA 13 Architecture Fix

CUDA 13.0 `nvcc` no longer accepts `compute_70`. PhysX 5.8.0's default GPU
architecture list includes it unless reduced GPU architectures are enabled.
Configure with `PX_GENERATE_GPU_REDUCED_ARCHITECTURES=ON` so the generated
`ARCH_CODE_LIST` starts at `compute_80` and includes `compute_89`.

This is the minimal direct CMake invocation used for KangEngine:

```bash
export PHYSX_ROOT=/home/asaid/Physics/PhysX/physx
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

### CUDA 13 `cuCtxCreate` Patch

PhysX 5.8.0 calls the older 3-argument CUDA Driver API form:

```cpp
cuCtxCreate(&mCtx, (unsigned int)flags, mDevHandle);
```

With CUDA 13 headers, `cuCtxCreate` maps to `cuCtxCreate_v4`, which expects a
`CUctxCreateParams*` argument. Patch
`/home/asaid/Physics/PhysX/physx/source/cudamanager/src/CudaContextManager.cpp`
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

### Verify Native Ada GPU Kernels

Check that the rebuilt GPU library contains `sm_89` cubins:

```bash
/usr/local/cuda/bin/cuobjdump --list-elf \
  /home/asaid/Physics/PhysX/physx/bin/linux.x86_64/release/libPhysXGpu_64.so \
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
/home/asaid/Physics/PhysX/physx/bin/linux.x86_64/release
```

</details>

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

3. If you build OpenUSD somewhere else, pass `-Dpxr_DIR=/path/to/usd_build` to your CMake configure command instead of modifying `CMakeLists.txt`.

## Python Bindings Optional

KangEngine exposes a Python module, `kangengine`, via pybind11. The extension is built by CMake and must be compiled against the same Python that will run it.

1. Create a virtual environment with Python 3.12 using `uv`.

    ```bash
    cd python
    uv venv .venv --python 3.12
    source .venv/bin/activate
    ```

2. Build the extension from the repo root.

    ```bash
    cd ..
    make build_python
    ```

    Or with USD support:

    ```bash
    make build_usd_python
    ```

3. Install the Python package in editable mode.

    ```bash
    cd python
    uv pip install -e .
    ```

4. Run an example.

    ```bash
    python examples/view_bvh_character.py
    ```

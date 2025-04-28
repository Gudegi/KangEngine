# KangEngine


## How to build 
I assume that you already installed basic tools like cmake.

dependencies
- OpenUSD 

    1. Clone https://github.com/PixarAnimationStudios/OpenUSD.git at home directory.
    2. mkdir usd_build in home directory.
    3. Build "python3 OpenUSD/build_scripts/build_usd.py ./usd_build"
    4. Place built usd folder at ~/usd_build or change set(pxr_DIR "$ENV{HOME}/usd_build") to the appropriate path.

- PhysX (only support version 5.1/tested only Mac)

    1. Clone https://github.com/o3de/PhysX.git (104.1 branch)
    2. brew install coreutils
    3. Navigate to physx/buildtools/packman
    4. Run ./packman update -y
    5. Makefiles are generated through a script in physx root directory: generate_projects.sh
    6. Run cmake --build . --config (debug|checked|profile|release) in compiler/mac-arm64.

install vckpg
- [git clone https://github.com/microsoft/vcpkg.git](https://learn.microsoft.com/ko-kr/vcpkg/get_started/get-started?pivots=shell-bash)

### Linux
1. cmake --preset=vcpkg
2. cmake --build build

### OSX(tested intel mac, M4 mac)
1. brew install ninja
1.2 Optional : brew install autoconf automake autoconf-archive
2. cmake --preset=vcpkg
3. cmake --build build


## Features
1. python bind support. (set IS_PYTHON_LIB to true in CMakeLists.txt before build.)


## TODO
1. Works only python 3.11
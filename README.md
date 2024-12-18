# KangEngine


## How to build 
I assume that you already installed basic tools like cmake.

dependencies
- OpenUSD 

1. https://github.com/PixarAnimationStudios/OpenUSD.git
2. Build
3. Place built usd folder at ~/usd_build or change set(pxr_DIR "$ENV{HOME}/usd_build") to the appropriate path.

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
1. Works only python 3.11.11.

aa
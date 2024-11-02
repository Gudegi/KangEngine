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

### OSX(tested only intel mac)
1. brew install ninja
2. cmake --preset=vcpkg
3. cmake --build build
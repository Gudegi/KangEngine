
run:
	./build/kangEngine

build:
	cmake --preset=vcpkg
	cmake --build build

debug:
	cmake --preset=vcpkg -DCMAKE_BUILD_TYPE=Debug
	cmake --build build

build_usd:
	cmake --preset=vcpkg -DUSE_USD=ON
	cmake --build build

build_python:
	cmake --preset=vcpkg -DIS_PYTHON_LIB=ON
	cmake --build build

build_usd_python:
	cmake --preset=vcpkg -DUSE_USD=ON -DIS_PYTHON_LIB=ON
	cmake --build build

clean_all:
	rm -rf ./build


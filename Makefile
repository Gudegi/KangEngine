
run:
	./build/kangEngine

build:
	cmake --preset=vcpkg
	cmake --build build

debug:
	cmake --preset=vcpkg -DCMAKE_BUILD_TYPE=Debug
	cmake --build build

clean_all:
	rm -rf ./build


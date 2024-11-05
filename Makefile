
run:
	./build/kangEngine

build:
	cmake --preset=vcpkg
	cmake --build build

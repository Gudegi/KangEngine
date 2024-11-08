
run:
	./build/kangEngine

build:
	cmake --preset=vcpkg
	cmake --build build

clean_all:
	rm -rf ./build


# .PHONY: build build_debug build_release build_performance run run_debug run_release run_performance clean_all

# Default build (Release)
build:
	cmake --preset=vcpkg
	cmake --build build/release
	@cp -f build/release/compile_commands.json build/compile_commands.json 2>/dev/null || true

# Debug build
build_debug:
	cmake --preset=vcpkg-debug
	cmake --build build/debug
	@cp -f build/debug/compile_commands.json build/compile_commands.json 2>/dev/null || true

# Release build (explicit) - same as default build
build_release: build

# Performance build (optimized + debug symbols for profiling)
build_performance:
	cmake --preset=vcpkg-performance
	cmake --build build/performance
	@cp -f build/performance/compile_commands.json build/compile_commands.json 2>/dev/null || true

# USD builds
build_usd:
	cmake --preset=vcpkg -DUSE_USD=ON
	cmake --build build/release

build_usd_debug:
	cmake --preset=vcpkg-debug -DUSE_USD=ON
	cmake --build build/debug

# Python builds
build_python:
	cmake --preset=vcpkg -DIS_PYTHON_LIB=ON
	cmake --build build/release

build_python_debug:
	cmake --preset=vcpkg-debug -DIS_PYTHON_LIB=ON
	cmake --build build/debug

# USD + Python builds
build_usd_python:
	cmake --preset=vcpkg -DUSE_USD=ON -DIS_PYTHON_LIB=ON
	cmake --build build/release

build_usd_python_debug:
	cmake --preset=vcpkg-debug -DUSE_USD=ON -DIS_PYTHON_LIB=ON
	cmake --build build/debug

# Run commands
run:
	./build/release/kangEngine

run_debug:
	./build/debug/kangEngine

run_release:
	./build/release/kangEngine

run_performance:
	./build/performance/kangEngine

# Clean commands
clean_all:
	rm -rf ./build

clean_debug:
	rm -rf ./build/build-debug

clean_release:
	rm -rf ./build/build-release

clean_performance:
	rm -rf ./build/build-performance


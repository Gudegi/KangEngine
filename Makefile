.PHONY: build build_debug build_release build_performance \
        build_usd build_usd_debug build_python build_python_debug \
        build_usd_python build_usd_python_debug \
        run run_debug run_release run_performance \
        clean_all clean_debug clean_release clean_performance

BUILD_DIR := build
RELEASE_DIR := $(BUILD_DIR)/release
DEBUG_DIR := $(BUILD_DIR)/debug
PERF_DIR := $(BUILD_DIR)/performance
EXECUTABLE := kangEngine

# configure + build + copy compile_commands
# $(1): preset, $(2): build dir, $(3): extra cmake flags
define do_build
	cmake --preset=$(1) $(3)
	cmake --build $(2)
	@cp -f $(2)/compile_commands.json $(BUILD_DIR)/compile_commands.json 2>/dev/null || true
endef

# Default build (Release)
build:
	$(call do_build,vcpkg,$(RELEASE_DIR),)

# Debug build
build_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),)

# Release build (explicit)
build_release: build

# Performance build
build_performance:
	$(call do_build,vcpkg-performance,$(PERF_DIR),)

# USD builds
build_usd:
	$(call do_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=ON)

build_usd_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),-DUSE_USD=ON)

# Python builds
build_python:
	$(call do_build,vcpkg,$(RELEASE_DIR),-DIS_PYTHON_LIB=ON)

build_python_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),-DIS_PYTHON_LIB=ON)

# USD + Python builds
build_usd_python:
	$(call do_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=ON -DIS_PYTHON_LIB=ON)

build_usd_python_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),-DUSE_USD=ON -DIS_PYTHON_LIB=ON)

# Run commands
run: run_release

run_debug:
	./$(DEBUG_DIR)/$(EXECUTABLE)

run_release:
	./$(RELEASE_DIR)/$(EXECUTABLE)

run_performance:
	./$(PERF_DIR)/$(EXECUTABLE)

# Clean commands
clean_all:
	rm -rf ./$(BUILD_DIR)

clean_debug:
	rm -rf ./$(DEBUG_DIR)

clean_release:
	rm -rf ./$(RELEASE_DIR)

clean_performance:
	rm -rf ./$(PERF_DIR)

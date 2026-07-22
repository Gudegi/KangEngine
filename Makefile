.PHONY: build build_all build_cuda build_current build_debug build_release build_relWithDebInfo \
        build_usd build_usd_debug build_python build_python_debug \
        build_python_cuda build_usd_python build_usd_python_debug wheel wheel_cuda \
        validate_physx_gpu validate_physx_gpu_cpp validate_sim_visual_batch \
        validate_python_api validate_render_component validate_wheel validate_wheel_cuda \
        docs docs_clean \
        run run2 run_debug run_release run_relWithDebInfo \
        clean_all clean_debug clean_release clean_relWithDebInfo

BUILD_DIR := build
RELEASE_DIR := $(BUILD_DIR)/release
DEBUG_DIR := $(BUILD_DIR)/debug
REL_DIR := $(BUILD_DIR)/relWithDebInfo
EXECUTABLE := KangEngine
PYTHON ?= python/.venv/bin/python
UV ?= uv
UNAME_S := $(shell uname -s)
DEFAULT_CMAKE_FLAGS := -DUSE_CUDA_INTEROP=OFF
CUDA_TOOLKIT_ROOT ?= /usr/local/cuda
PHYSX_CUDA_BIN_PLATFORM ?= linux.x86_64
CUDA_INTEROP_CMAKE_FLAGS := -DUSE_CUDA_INTEROP=ON -DCUDAToolkit_ROOT=$(CUDA_TOOLKIT_ROOT) -DCUDAToolkit_NVCC_EXECUTABLE=$(CUDA_TOOLKIT_ROOT)/bin/nvcc -DPHYSX_BIN_PLATFORM=$(PHYSX_CUDA_BIN_PLATFORM)

# Easy workflow
# 1. make build_all
# 2. make run2

# configure + build + copy compile_commands
# $(1): preset, $(2): build dir, $(3): extra cmake flags
define do_build
	cmake --preset=$(1) $(DEFAULT_CMAKE_FLAGS) $(3)
	cmake --build $(2)
	@cp -f $(2)/compile_commands.json $(BUILD_DIR)/compile_commands.json 2>/dev/null || true
endef

define do_cuda_build
	cmake --preset=$(1) $(CUDA_INTEROP_CMAKE_FLAGS) $(3)
	cmake --build $(2)
	@cp -f $(2)/compile_commands.json $(BUILD_DIR)/compile_commands.json 2>/dev/null || true
endef

# Default build (Release)
build:
	$(call do_build,vcpkg,$(RELEASE_DIR),)

ifeq ($(UNAME_S),Linux)
build_all:
	$(call do_cuda_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=ON -DIS_PYTHON_LIB=ON)
else
build_all:
	$(call do_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=ON -DIS_PYTHON_LIB=ON)
endif

build_cuda:
	$(call do_cuda_build,vcpkg,$(RELEASE_DIR),)

build_current:
	cmake --build $(RELEASE_DIR)

# Debug build
build_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),)

# Release build (explicit)
build_release: build

# RelWithDebInfo build
build_relWithDebInfo:
	$(call do_build,vcpkg-relWithDebInfo,$(REL_DIR),)

# USD builds
build_usd:
	$(call do_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=ON)

build_usd_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),-DUSE_USD=ON)

# Python builds
build_python:
	$(call do_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=OFF -DIS_PYTHON_LIB=ON)

build_python_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),-DIS_PYTHON_LIB=ON)

build_python_cuda:
	$(call do_cuda_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=OFF -DIS_PYTHON_LIB=ON)

validate_physx_gpu: build_python_cuda
	PYTHONPATH=python $(PYTHON) python/examples/smoke/physics_gpu_system_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/kangsimworld_gpu_root_state_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/kangsimworld_gpu_articulation_state_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/kangsimworld_gpu_articulation_control_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/kangsimworld_gpu_contact_sensor_smoke.py

validate_physx_gpu_cpp: build_cuda
	./$(RELEASE_DIR)/physx_gpu_step_smoke

validate_sim_visual_batch: build_python
	PYTHONPATH=python $(PYTHON) python/examples/smoke/sim_visual_batch_smoke.py

validate_python_api: build_python
	PYTHONPYCACHEPREFIX=/tmp/kangengine-pycache $(PYTHON) -m py_compile \
		python/kangengine/__init__.py \
		python/kangengine/app/__init__.py \
		python/kangengine/app/application.py \
		python/kangengine/material/__init__.py \
		python/kangengine/material/materials.py \
		python/kangengine/physics/__init__.py \
		python/kangengine/physics/wrappers.py \
		python/kangengine/render/__init__.py \
		python/kangengine/sim/__init__.py \
		python/kangengine/sim/world.py \
		python/kangengine/sim/sensor.py \
		python/kangengine/terrain/__init__.py \
		python/kangengine/terrain/heightfield.py \
		python/kangengine/motion_module/__init__.py \
		python/kangengine/motion_module/editor.py \
		python/kangengine/motion_module/modules.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/public_api_surface_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/public_stub_surface_smoke.py

# TODO: USD support
# macOS
wheel: build_python
	$(PYTHON) python/scripts/validate_wheel.py --python $(PYTHON) --uv $(UV) \
		--expect-no-usd --build-only --output-dir python/dist
# Linux + cuda(13.0)
wheel_cuda: build_python_cuda
	$(PYTHON) python/scripts/validate_wheel.py --python $(PYTHON) --uv $(UV) \
		--expect-no-usd --build-only --output-dir python/dist

validate_wheel: build_python
	$(PYTHON) python/scripts/validate_wheel.py --python $(PYTHON) --uv $(UV) \
		--expect-no-usd

validate_wheel_cuda: build_python_cuda
	$(PYTHON) python/scripts/validate_wheel.py --python $(PYTHON) --uv $(UV) \
		--expect-no-usd

validate_render_component: build_python
	PYTHONPATH=python $(PYTHON) python/examples/smoke/obj_material_loader_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/scene_add_obj_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/mjcf_visual_rgba_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/mjcf_visual_duplicate_mesh_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/articulation_binding_component_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/debug_draw_transform_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/quaternion_binding_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/transform_component_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/mesh_component_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/resource_component_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/scene_resource_manager_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/python_resource_manager_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/render_component_lifecycle_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/scene_render_system_smoke.py
	PYTHONPATH=python $(PYTHON) python/examples/smoke/scene_render_instancing_smoke.py

# USD + Python builds
build_usd_python:
	$(call do_build,vcpkg,$(RELEASE_DIR),-DUSE_USD=ON -DIS_PYTHON_LIB=ON)

build_usd_python_debug:
	$(call do_build,vcpkg-debug,$(DEBUG_DIR),-DUSE_USD=ON -DIS_PYTHON_LIB=ON)

# Documentation
docs:
	uv run --project python --extra docs sphinx-build -b html _private/sphinx_docs _private/sphinx_docs/_build/html

docs_clean:
	rm -rf _private/sphinx_docs/_build

# Run commands
run: run_release

# Build and run
run2: build_current
	$(MAKE) run_release

run_debug:
	./$(DEBUG_DIR)/$(EXECUTABLE)

run_release:
	./$(RELEASE_DIR)/$(EXECUTABLE)

run_relWithDebInfo:
	./$(REL_DIR)/$(EXECUTABLE)

# Clean commands
clean_all:
	rm -rf ./$(BUILD_DIR)

clean_debug:
	rm -rf ./$(DEBUG_DIR)

clean_release:
	rm -rf ./$(RELEASE_DIR)

clean_relWithDebInfo:
	rm -rf ./$(REL_DIR)

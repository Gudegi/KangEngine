cmake_minimum_required(VERSION 3.16)
project(KangEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For vscode-clangd

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug, Release, Performance)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# Compiler flags for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DKE_DEBUG")
    message(STATUS "Debug build: Debug symbols enabled, optimizations disabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DKE_NDEBUG")
    message(STATUS "Release build: Full optimizations, no debug symbols")
elseif(CMAKE_BUILD_TYPE STREQUAL "Performance")
    set(CMAKE_CXX_FLAGS_Performance "-O2 -g -DKE_NDEBUG")
    message(STATUS "Release with debug info: Optimizations + debug symbols")
endif()

set(VCPKG_BUILD_TYPE release)
set(pxr_DIR "$ENV{HOME}/usd_build") # User config path
set(physx_DIR "$ENV{HOME}/Physics/PhysX/physx") # User config path

set(ASSETS_DIR "${CMAKE_SOURCE_DIR}/assets")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(MAIN_DIR "${CMAKE_SOURCE_DIR}/examples")
set(BIND_DIR "${CMAKE_SOURCE_DIR}/bindings")
option(IS_PYTHON_LIB "Build Python bindings" OFF) # Use -DIS_PYTHON_LIB=ON to enable
option(USE_USD "Enable USD support" OFF) # Use -DUSE_USD=ON to enable

if(USE_USD)
    if(NOT EXISTS "${pxr_DIR}")
        message(FATAL_ERROR "pxr_DIR path '${pxr_DIR}' does not exist. Please check your USD build path.")
    endif()

    # USD 경로 설정
    set(CMAKE_PREFIX_PATH "${pxr_DIR}" ${CMAKE_PREFIX_PATH})
    find_package(pxr REQUIRED)

    add_compile_definitions(KANGENGINE_USE_USD)

    # TBB macros required for USD headers
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_definitions(TBB_USE_ASSERT=0)
        add_compile_definitions(TBB_USE_THREADING_TOOLS=0)
        add_compile_definitions(TBB_USE_PERFORMANCE_WARNINGS=0)
    endif()

    # USD include 경로 추가
    include_directories(${pxr_DIR}/include)
    link_directories(${pxr_DIR}/lib)

    message(STATUS "USD support enabled")
    message(STATUS "USD include dir: ${pxr_DIR}/include")
    message(STATUS "USD lib dir: ${pxr_DIR}/lib")
endif()
if(NOT EXISTS "${physx_DIR}")
    message(FATAL_ERROR "physx_DIR path '${physx_DIR}' does not exist. Please check your Physx build path.")
else()
    add_compile_definitions(KANGENGINE_USE_PHYSX)
    message(STATUS "PhysX directory found. Linking PhysX libraries.")
    #set(PHYSX_BUILD_TYPE "checked" CACHE STRING "The build type of PhysX")
    set(PHYSX_BUILD_TYPE "release" CACHE STRING "The build type of PhysX")
    set_property(CACHE PHYSX_BUILD_TYPE PROPERTY STRINGS debug checked profile release)
    if(NOT CMAKE_BUILD_TYPE)
        if(PHYSX_BUILD_TYPE STREQUAL "debug" OR PHYSX_BUILD_TYPE STREQUAL "checked")
            set(CMAKE_BUILD_TYPE "Debug")
        endif()
    endif()
endif()

#add_definitions(-DSTB_IMAGE_IMPLEMENTATION)
# STB_IMAGE_IMPLEMENTATION is defined only in src/backend/base/base_utils.cpp
# Do not define it globally to avoid duplicate symbol errors

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
if(IS_PYTHON_LIB)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()
if(EXISTS "${physx_DIR}")
    include_directories(${physx_DIR}/include)
    if(APPLE)
        link_directories("${physx_DIR}/bin/mac.x86_64/${PHYSX_BUILD_TYPE}") # TODO: actually it need to be not x86 but arm. fix build setting
    elseif(UNIX)
        link_directories("${physx_DIR}/bin/linux.x86_64/${PHYSX_BUILD_TYPE}") # This is the path where PhysX libraries are installed
    else()
        message("Not support OS setting for PhysX")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Building project in debug with PhysX ${PHYSX_BUILD_TYPE} configuration")
        add_compile_definitions(_DEBUG)
    else()
        message("Building project in release configuration with PhysX ${PHYSX_BUILD_TYPE} configuration")
        add_compile_definitions(NDEBUG)
    endif()
endif()

file(GLOB_RECURSE SOURCE_FILES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE HEADER_FILES "${SRC_DIR}/*.hpp")
file(GLOB_RECURSE MAIN_FILES "${MAIN_DIR}/*.cpp")

# Executable for KangEngine
#add_executable(${PROJECT_NAME} ${MAIN_FILES} ${SOURCE_FILES})
#file(GLOB MAIN_FILE "${MAIN_DIR}/standalone_physx_hello.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/backend_light0.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/backend_phong_light.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/backend_material.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/physx_drop_box.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/test_obj.cpp")
file(GLOB MAIN_FILE "${MAIN_DIR}/test_prim_scene.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/usd_example.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/scene_graph_example.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/scene_usd_example.cpp")
add_executable(${PROJECT_NAME} ${MAIN_FILE} ${SOURCE_FILES})
add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSETS_DIR}" "${CMAKE_BINARY_DIR}/../assets")
add_dependencies(${PROJECT_NAME} copy_assets)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${SRC_DIR}
    ${Stb_INCLUDE_DIR}
)

# Link libraries for KangEngine
target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    glad::glad
    glm::glm
    imgui::imgui
    imguizmo::imguizmo
    fmt::fmt
    tinyobjloader::tinyobjloader
)

# Link USD
if(USE_USD)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        usd
        usdGeom
        sdf
        tf
    )
endif()

if(EXISTS "${physx_DIR}")
    if(APPLE)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            PhysXExtensions_static_64
            PhysX_static_64
            PhysXPvdSDK_static_64
            PhysXCommon_static_64
            PhysXFoundation_static_64
            pthread
            dl
        )
    elseif(UNIX)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            -Wl,--start-group
            PhysXExtensions_static_64
            PhysX_static_64
            PhysXPvdSDK_static_64
            PhysXCommon_static_64
            PhysXFoundation_static_64
            -Wl,--end-group
            pthread
            dl
        )
    endif()
endif()

# Python Module for pybind11
if(IS_PYTHON_LIB)
    set(PYTHON_MODULE_NAME "kangengine")
    #file(GLOB_RECURSE PYBIND_SOURCE_FILES "${BIND_DIR}/*.cpp")
    file(GLOB PYBIND_SOURCE_FILES
        "${BIND_DIR}/bind_new.cpp"
        "${BIND_DIR}/bind_scene.cpp"
    )
    message(STATUS "PY_SOURCE: ${PYBIND_SOURCE_FILES}")
    
    Python_add_library(${PYTHON_MODULE_NAME} 
        ${PYBIND_SOURCE_FILES}
        ${SOURCE_FILES}
        ${HEADER_FILES}
    )
    target_include_directories(${PYTHON_MODULE_NAME} PRIVATE
        ${SRC_DIR}
        ${Stb_INCLUDE_DIR}
    )
    target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE
        pybind11::headers
        glfw
        glad::glad
        glm::glm
        imgui::imgui
        imguizmo::imguizmo
    )

    # Link USD libraries for Python module
    if(USE_USD)
        target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE
            usd
            usdGeom
            sdf
            tf
        )
    endif()

    # Link PhysX libraries for Python module
    if(EXISTS "${physx_DIR}")
        if(APPLE)
            target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE
                PhysXExtensions_static_64
                PhysX_static_64
                PhysXPvdSDK_static_64
                PhysXCommon_static_64
                PhysXFoundation_static_64
                pthread
                dl
            )
        elseif(UNIX)
            target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE
                -Wl,--start-group
                PhysXExtensions_static_64
                PhysX_static_64
                PhysXPvdSDK_static_64
                PhysXCommon_static_64
                PhysXFoundation_static_64
                -Wl,--end-group
                pthread
                dl
            )
        endif()
    endif()
    set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN ON
    )

    install(TARGETS ${PYTHON_MODULE_NAME}
        LIBRARY  DESTINATION ${BIND_DIR}
    )
endif()
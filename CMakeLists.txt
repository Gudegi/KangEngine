cmake_minimum_required(VERSION 3.16)
project(KangEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For vscode-clangd

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug, Release, RelWithDebInfo)" FORCE)
endif()
message(STATUS "----Build Start, type: ${CMAKE_BUILD_TYPE}")

# Compiler flags for different build types
add_library(build_configs INTERFACE)
target_compile_definitions(build_configs INTERFACE
    $<$<CONFIG:Debug>:KE_DEBUG _DEBUG>
    $<$<CONFIG:Release>:KE_NDEBUG NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:KE_NDEBUG NDEBUG>
)
target_compile_options(build_configs INTERFACE
    $<$<CONFIG:Debug>:-O0 -g>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
)

set(VCPKG_BUILD_TYPE release)
set(ASSETS_DIR "${CMAKE_SOURCE_DIR}/assets")
set(ASSETS_SIMLINK_DIR "${CMAKE_BINARY_DIR}/../assets")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(MAIN_DIR "${CMAKE_SOURCE_DIR}/examples")
set(BIND_DIR "${CMAKE_SOURCE_DIR}/bindings")

option(IS_PYTHON_LIB "Build Python bindings" OFF) # Use -DIS_PYTHON_LIB=ON to enable
option(USE_USD "Enable USD support" OFF) # Use -DUSE_USD=ON to enable
option(USE_PHYSX "Enable Physx support" ON) # Use -DUSE_PHYSX=ON to enable


##### Link pre-compiled libs #####
add_library(pre_compiled_ext INTERFACE)

if(USE_USD)
    set(pxr_DIR "$ENV{HOME}/usd_build") # User config path
    if(NOT EXISTS "${pxr_DIR}")
        message(FATAL_ERROR "pxr_DIR path '${pxr_DIR}' does not exist. Please check your USD build path.")
    endif()
    set(CMAKE_PREFIX_PATH "${pxr_DIR}" ${CMAKE_PREFIX_PATH})
    find_package(pxr REQUIRED)

    target_compile_definitions(pre_compiled_ext INTERFACE KANGENGINE_USE_USD)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(pre_compiled_ext INTERFACE 
            TBB_USE_ASSERT=0 
            TBB_USE_THREADING_TOOLS=0 
            TBB_USE_PERFORMANCE_WARNINGS=0
        )
    endif()

    target_include_directories(pre_compiled_ext INTERFACE ${pxr_DIR}/include)
    target_link_directories(pre_compiled_ext INTERFACE ${pxr_DIR}/lib)
    target_link_libraries(pre_compiled_ext INTERFACE
        usd
        usdGeom
        sdf
        tf
    )

    message(STATUS "USD support enabled")
    message(STATUS "USD include dir: ${pxr_DIR}/include")
    message(STATUS "USD lib dir: ${pxr_DIR}/lib")
endif()

if(USE_PHYSX)
    set(physx_DIR "$ENV{HOME}/Physics/PhysX/physx") # User config path
    if(NOT EXISTS "${physx_DIR}")
        message(FATAL_ERROR "physx_DIR path '${physx_DIR}' does not exist. Please check your Physx build path.")
    endif()
    
    target_compile_definitions(pre_compiled_ext INTERFACE KANGENGINE_USE_PHYSX)
    #set(PHYSX_BUILD_TYPE "checked" CACHE STRING "The build type of PhysX")
    set(PHYSX_BUILD_TYPE "release" CACHE STRING "The build type of PhysX")
    set_property(CACHE PHYSX_BUILD_TYPE PROPERTY STRINGS debug checked profile release)
    if(NOT CMAKE_BUILD_TYPE)
        if(PHYSX_BUILD_TYPE STREQUAL "debug" OR PHYSX_BUILD_TYPE STREQUAL "checked")
            set(CMAKE_BUILD_TYPE "Debug")
        endif()
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("Building project in debug with PhysX ${PHYSX_BUILD_TYPE} configuration")
        target_compile_definitions(pre_compiled_ext INTERFACE _DEBUG)
    else()
        message("Building project in release configuration with PhysX ${PHYSX_BUILD_TYPE} configuration")
        target_compile_definitions(pre_compiled_ext INTERFACE NDEBUG)
    endif()

    target_include_directories(pre_compiled_ext INTERFACE ${physx_DIR}/include)
    if(APPLE)
        target_link_directories(pre_compiled_ext INTERFACE "${physx_DIR}/bin/mac.x86_64/${PHYSX_BUILD_TYPE}") # TODO: actually it need to be not x86 but arm. fix build setting
    elseif(UNIX)
        target_link_directories(pre_compiled_ext INTERFACE "${physx_DIR}/bin/linux.x86_64/${PHYSX_BUILD_TYPE}") # This is the path where PhysX libraries are installed
    else()
        message(FATAL_ERROR "Not support OS setting for PhysX")
    endif()

    if(APPLE)
        target_link_libraries(pre_compiled_ext INTERFACE
            PhysXExtensions_static_64
            PhysX_static_64
            PhysXPvdSDK_static_64
            PhysXCommon_static_64
            PhysXFoundation_static_64
            pthread
            dl
        )
    elseif(UNIX)
        target_link_libraries(pre_compiled_ext INTERFACE
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
##### Link pre-compiled libs END #####

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
find_package(Eigen3 CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
if(IS_PYTHON_LIB)
    # Prefer system Python over vcpkg's Python
    set(Python_FIND_STRATEGY LOCATION)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")
    find_package(pybind11 CONFIG REQUIRED)
endif()


###### core libs ######
file(GLOB_RECURSE SOURCE_FILES "${SRC_DIR}/*.cpp")
add_library(engine_core STATIC ${SOURCE_FILES})
target_include_directories(engine_core PUBLIC
    ${SRC_DIR}
    ${Stb_INCLUDE_DIR}
)
target_link_libraries(engine_core PUBLIC
    build_configs
    glfw
    glad::glad
    glm::glm
    imgui::imgui
    imguizmo::imguizmo
    fmt::fmt
    tinyobjloader::tinyobjloader
    Eigen3::Eigen
    tinyxml2::tinyxml2
)
target_link_libraries(engine_core PUBLIC pre_compiled_ext)
###### core libs END ######

# Executable for KangEngine
#file(GLOB MAIN_FILE "${MAIN_DIR}/standalone_physx_hello.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/backend_light0.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/backend_phong_light.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/backend_material.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/physx_drop_box.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/test_obj.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/test_prim_scene.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/usd_example.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/scene_graph_example.cpp")
#file(GLOB MAIN_FILE "${MAIN_DIR}/scene_usd_example.cpp")
# Aboves are legacy.
#file(GLOB MAIN_FILE "${MAIN_DIR}/test_stl.cpp")
file(GLOB MAIN_FILE "${MAIN_DIR}/test_xml.cpp")
add_executable(${PROJECT_NAME} ${MAIN_FILE})
target_link_libraries(${PROJECT_NAME} PRIVATE engine_core)

# Define asset path macro for C++ executable
target_compile_definitions(${PROJECT_NAME} PRIVATE
    KANGENGINE_ASSETS_ROOT="${CMAKE_SOURCE_DIR}/assets"
)

# Create asset folder simlink
if(EXISTS "${ASSETS_SIMLINK_DIR}")
    file(REMOVE_RECURSE "${ASSETS_SIMLINK_DIR}")
endif()
file(CREATE_LINK 
    "${ASSETS_DIR}"
    "${ASSETS_SIMLINK_DIR}"
    SYMBOLIC
)

# Python Module for pybind11
if(IS_PYTHON_LIB)
    set(PYTHON_MODULE_NAME "_kangengine")
    set(PYTHON_DL_DIR "${CMAKE_SOURCE_DIR}/python/${PROJECT_NAME}")
    #file(GLOB_RECURSE PYBIND_SOURCE_FILES "${BIND_DIR}/*.cpp")
    file(GLOB PYBIND_SOURCE_FILES
        "${BIND_DIR}/bind_new.cpp"
        "${BIND_DIR}/bind_scene.cpp"
    )
    message(STATUS "PY_SOURCE: ${PYBIND_SOURCE_FILES}")
    
    Python_add_library(${PYTHON_MODULE_NAME} MODULE
        ${PYBIND_SOURCE_FILES}
    )
    # Define asset path macro for python lib
    target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE
        KANGENGINE_ASSETS_ROOT="${PYTHON_DL_DIR}/assets"
    )
    
    target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE
        engine_core
        pybind11::headers   
    )

    set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY "${PYTHON_DL_DIR}"  # Unix
        # RUNTIME_OUTPUT_DIRECTORY "${PYTHON_DL_DIR}"  # Windows
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN ON
    )

    set(PYTHON_ASSETS_DIR "${PYTHON_DL_DIR}/assets")
    if(NOT EXISTS "${PYTHON_ASSETS_DIR}")
        file(CREATE_LINK "${ASSETS_DIR}" "${PYTHON_ASSETS_DIR}" SYMBOLIC)
    endif()
    #add_custom_command(TARGET ${PYTHON_MODULE_NAME} POST_BUILD
    #    COMMAND ${CMAKE_COMMAND} -E copy_directory
    #    "${ASSETS_DIR}" "PYTHON_ASSETS_DIR"
    #    COMMENT "Copying assets to Python module directory"
    #)

    # cmake --build build/release --target install
    install(DIRECTORY "${ASSETS_DIR}/" 
        DESTINATION "${PYTHON_ASSETS_DIR}")
    install(TARGETS ${PYTHON_MODULE_NAME}
        LIBRARY  DESTINATION ${PYTHON_DL_DIR}
    )
endif()
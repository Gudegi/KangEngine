cmake_minimum_required(VERSION 3.16)
project(KangEngine)

set(CMAKE_CXX_STANDARD 17)
set(VCPKG_BUILD_TYPE release)
set(pxr_DIR "$ENV{HOME}/usd_build")
set(ASSETS_DIR "${CMAKE_SOURCE_DIR}/assets")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(MAIN_DIR "${CMAKE_SOURCE_DIR}/examples")
set(BIND_DIR "${CMAKE_SOURCE_DIR}/bindings")
set(IS_PYTHON_LIB true)

add_definitions(-DSTB_IMAGE_IMPLEMENTATION)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE SOURCE_FILES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE HEADER_FILES "${SRC_DIR}/*.hpp")
file(GLOB_RECURSE MAIN_FILES "${MAIN_DIR}/*.cpp")

# Executable for KangEngine
add_executable(${PROJECT_NAME} ${MAIN_FILES} ${SOURCE_FILES})
add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSETS_DIR}" "${CMAKE_BINARY_DIR}/assets")
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
)

# Python Module for pybind11
if(IS_PYTHON_LIB)
    set(PYTHON_MODULE_NAME "kangengine")
    file(GLOB_RECURSE PYBIND_SOURCE_FILES "${BIND_DIR}/*.cpp")
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
    set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN ON
    )

    install(TARGETS ${PYTHON_MODULE_NAME}
        LIBRARY  DESTINATION ${BIND_DIR}
    )
endif()
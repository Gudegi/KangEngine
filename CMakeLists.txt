# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(KangEngine)

set(CMAKE_CXX_STANDARD 17)

set(pxr_DIR "$ENV{HOME}/usd_build")
set(ASSETS_DIR "${CMAKE_SOURCE_DIR}/assets")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

add_definitions(-DSTB_IMAGE_IMPLEMENTATION)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
#find_package(pxr REQUIRED)

file(GLOB_RECURSE SOURCE_FILES "${SRC_DIR}/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${ASSETS_DIR}" "${CMAKE_BINARY_DIR}/assets")
add_dependencies(${PROJECT_NAME} copy_assets)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${SRC_DIR}
    ${Stb_INCLUDE_DIR}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    glad::glad
    glm::glm
    imgui::imgui
    imguizmo::imguizmo
    #${PXR_LIBRARIES}
    )
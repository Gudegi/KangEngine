# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(KangEngine)

set(CMAKE_CXX_STANDARD 17)

set(pxr_DIR "$ENV{HOME}/usd_build")

add_definitions(-DSTB_IMAGE_IMPLEMENTATION)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(cglm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
#find_package(pxr REQUIRED)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ./src/main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
     ${Stb_INCLUDE_DIR}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    glad::glad
    cglm::cglm
    imgui::imgui
    imguizmo::imguizmo
    #${PXR_LIBRARIES}
    )
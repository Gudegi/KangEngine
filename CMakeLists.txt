# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(KangEngine)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
#find_package(glm REQUIRED)

add_executable(kangEngine src/main.cpp)

target_link_libraries(
    kangEngine
    glfw
    glad::glad)

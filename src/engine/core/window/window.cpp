#include "window.hpp"
#include "engine/core/app/app.hpp"
#include "utils/asset_path.hpp"
#include <fmt/base.h>
#include <fstream>
#include <stdexcept>

namespace KE {

Window::Window() {
    _width = 1920;
    _height = 1080;
    _logicalWidth = _width;
    _logicalHeight = _height;
    this->initGlfw();
    this->initGlad(); // TODO: Remove OpenGL dependency
}

Window::~Window() { close(); }

void Window::close() {
    if (!_window)
        return;
    glfwDestroyWindow(_window);
    _window = nullptr;
    glfwTerminate();
}

void Window::init(int width, int height, bool headless) {
    if (_window == nullptr)
        return;

    _width = width;
    _height = height;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) {
        monitor = glfwGetPrimaryMonitor();
    }
    int windowWidth = _width;
    int windowHeight = _height;
#ifdef __APPLE__
    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    if (xscale > 0.0f && yscale > 0.0f) {
        windowWidth = static_cast<int>(_width / xscale);
        windowHeight = static_cast<int>(_height / yscale);
    }
#endif
    glfwSetWindowSize(_window, windowWidth, windowHeight);
    if (headless)
        glfwHideWindow(_window);
    else
        glfwShowWindow(_window);
    // Read back actual logical size (differ due to DPI scaling)
    glfwGetWindowSize(_window, &_logicalWidth, &_logicalHeight);
    glViewport(0, 0, _width, _height); // TODO: Remove OpenGL dependency
}

void Window::initGlfw() {
#ifdef __APPLE__
    // Prevent changing current directory
    glfwInitHint(GLFW_COCOA_CHDIR_RESOURCES, GLFW_FALSE);
#endif
    if (glfwInit() != GLFW_TRUE)
        throw std::runtime_error("Failed to initialize GLFW");
    std::ifstream mappingFile(getAssetPath("gamepad/gamecontrollerdb.txt"));
    if (mappingFile) {
        const std::string mappings{
            std::istreambuf_iterator<char>(mappingFile),
            std::istreambuf_iterator<char>()};
        if (!mappings.empty() &&
            glfwUpdateGamepadMappings(mappings.c_str()) != GLFW_TRUE) {
            fmt::println(stderr, "Failed to load gamepad mappings");
        }
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    _window = glfwCreateWindow(_width, _height, "KangEngine", nullptr, nullptr);
    if (_window == NULL) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(_window);
    // glfwSetWindowUserPointer(_window, this);
    setVSync(true);
}

void Window::setVSync(bool enabled) {
    _vsync = enabled;
    if (_window)
        glfwSwapInterval(enabled ? 1 : 0);
}

void Window::initGlad() {
    if (_window == nullptr)
        return;

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        close();
        throw std::runtime_error("Failed to initialize GLAD");
    }
}

} // namespace KE

///
/// UIScale — cached UI scale for panels and widgets.
///

#ifndef _UI_SCALE_HPP_
#define _UI_SCALE_HPP_

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>

namespace KE {

class UIScale {
  public:
    // Refresh cached size/DPI values from a GLFW window.
    bool update(GLFWwindow* window) {
        if (!window)
            return false;

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        int windowWidth = 0;
        int windowHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        const float dpiScale = monitorDpiScale(window);
        return update(framebufferWidth, framebufferHeight, windowWidth,
                      windowHeight, dpiScale);
    }

    // Refresh cached values from explicit framebuffer/window metrics.
    bool update(int framebufferWidth, int framebufferHeight, int windowWidth,
                int windowHeight, float dpiScale) {
        dpiScale = std::max(dpiScale, 1e-6f);
        const bool changed = framebufferWidth != _framebufferWidth ||
                             framebufferHeight != _framebufferHeight ||
                             windowWidth != _windowWidth ||
                             windowHeight != _windowHeight ||
                             std::abs(dpiScale - _dpiScale) > 1e-4f ||
                             std::abs(_userScale - _lastUserScale) > 1e-4f;
        if (!changed)
            return false;

        _framebufferWidth = framebufferWidth;
        _framebufferHeight = framebufferHeight;
        _windowWidth = windowWidth;
        _windowHeight = windowHeight;
        _dpiScale = dpiScale;
        _lastUserScale = _userScale;
        return true;
    }

    // Set the user-controlled UI scale multiplier.
    void setUserScale(float userScale) {
        _userScale = std::max(userScale, 1e-6f);
    }

    // Convert logical ImGui layout pixels using only the user scale.
    float logicalPx(float px) const { return px * _userScale; }
    // Convert logical pixels to framebuffer/device pixels.
    float devicePx(float px) const { return px * _dpiScale * _userScale; }
    // Current logical UI scale multiplier.
    float value() const { return _userScale; }
    // Cached monitor/content DPI scale.
    float dpiScale() const { return _dpiScale; }
    // User-controlled scale multiplier.
    float userScale() const { return _userScale; }

    // Cached framebuffer width in device pixels.
    int framebufferWidth() const { return _framebufferWidth; }
    // Cached framebuffer height in device pixels.
    int framebufferHeight() const { return _framebufferHeight; }
    // Cached window width in logical pixels.
    int windowWidth() const { return _windowWidth; }
    // Cached window height in logical pixels.
    int windowHeight() const { return _windowHeight; }

  private:
    // Read the active monitor content scale, falling back to primary monitor.
    static float monitorDpiScale(GLFWwindow* window) {
        GLFWmonitor* monitor = glfwGetWindowMonitor(window);
        if (!monitor)
            monitor = glfwGetPrimaryMonitor();
        if (!monitor)
            return 1.0f;

        float xScale = 1.0f;
        float yScale = 1.0f;
        glfwGetMonitorContentScale(monitor, &xScale, &yScale);
        return std::max((xScale + yScale) * 0.5f, 1e-6f);
    }

    float _dpiScale = 1.0f;
    float _userScale = 1.0f;
    float _lastUserScale = 1.0f;
    int _framebufferWidth = 0;
    int _framebufferHeight = 0;
    int _windowWidth = 0;
    int _windowHeight = 0;
};

} // namespace KE

#endif

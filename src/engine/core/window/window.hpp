#ifndef _WINDOW_HPP_
#define _WINDOW_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

namespace KE {

class Window

{

  private:
    int _width, _height;               // framebuffer pixels
    int _logicalWidth, _logicalHeight; // screen/logical pixels
    GLFWwindow* _window;
    void initGlfw();
    void initGlad();

  public:
    Window();
    ~Window();
    void init(int width, int height);
    GLFWwindow* getGlfwWindow() { return _window; }
    int getLogicalWidth() const { return _logicalWidth; }
    int getLogicalHeight() const { return _logicalHeight; }
};

} // namespace KE

#endif
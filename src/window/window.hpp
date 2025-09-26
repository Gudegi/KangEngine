#ifndef _WINDOW_HPP_
#define _WINDOW_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

namespace KE {

class Window

{

private:
    int _width, _height;
    GLFWwindow* _window;
    void initGlfw();
    void initGlad();

public:
    Window();
    ~Window();
    void init(int width, int height);
    GLFWwindow* getGlfwWindow() {return _window;}
};

} // namespace KE

#endif
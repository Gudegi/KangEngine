#ifndef _WINDOW_HPP_
#define _WINDOW_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

class Window

{

private:
    int _width, _height;
    GLFWwindow* _window;
    void initGlfw();
    void initGlad();
    static void framebufferSizeCallbackWrapper(GLFWwindow* window, int width, int height);
    static void scrollCallbackWrapper(GLFWwindow* window, double xoffset, double yoffset);
    static void cursorPositionCallbackWrapper(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods);

public:
    Window();
    Window(int width, int height);
    ~Window();
    GLFWwindow* getGlfwWindow() {return _window;}

};


#endif
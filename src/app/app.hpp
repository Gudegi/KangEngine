///
/// Author Kyungwon Kang, 2024/11
///

// TODO: fix me (separate class)

#ifndef _APP_HPP_
#define _APP_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "kangEngine.hpp"

class App
{

private:
    int _width, _height;
    bool _hideUi;
    GLFWwindow* initGlfw();
    void initGlad();
    Camera _camera = Camera(glm::vec3(0, 1, -1), glm::vec3(0, 0, 0));

    static void framebufferSizeCallbackWrapper(GLFWwindow* window, int width, int height);
    static void scrollCallbackWrapper(GLFWwindow* window, double xoffset, double yoffset);
    static void cursorPositionCallbackWrapper(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods);

public:
    App(int width, int height, bool hideUi);
    ~App();

    Camera& camera() { return _camera; }

    virtual void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    virtual void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    virtual void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    virtual void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);


};

#endif
///
/// Author Kyungwon Kang, 2024/11
///

// TODO: fix me (separate class)

#ifndef _APP_HPP_
#define _APP_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "camera/camera.hpp"
#include "window/window.hpp"
#include "ui/panel_manager.hpp"
#include <memory>
class App
{

private:
    int _width, _height;
    bool _hideUi;
    
    Window _window;
    Camera _camera;
    PanelManager _mainPanel = PanelManager(this->getWindow());
    //Light _light;

    void start();

public:
    App(int width, int height, bool hideUi);
    ~App();


    struct IO;
    struct RenderVariable;
    std::unique_ptr<App::IO> _io;
    std::unique_ptr<App::RenderVariable> _renderVariable;
    Camera& getCamera() { return _camera; }
    GLFWwindow* getWindow() { return _window.getGlfwWindow();}

    virtual void setUp() {} // 처음에 사용
    virtual void preRender() {} // 루프 안에서 사용됨. 렌더 전에 사용
    virtual void render() {} // 실제 렌더링
    virtual void postRender() {} // 렌더링 이후 마무리

    virtual void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    virtual void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    virtual void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    virtual void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);


};

#endif
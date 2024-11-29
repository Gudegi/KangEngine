///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _APP_HPP_
#define _APP_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "camera/camera.hpp"
#include "window/window.hpp"
#include "ui/panel_manager.hpp"
#include "mesh/mesh_manager.hpp"

#include <memory>
struct ShapeGlBuffer
{
    VAO* vao;
    VBO* vbo;
    EBO* ebo;
    int numTri;
};
class App
{
private:
    class GLFWCallbackWrapper
    {//https://stackoverflow.com/a/41089765
    private:
        static App* _app;
    public:
        GLFWCallbackWrapper() = delete;
        GLFWCallbackWrapper(const GLFWCallbackWrapper&) = delete;
        GLFWCallbackWrapper(GLFWCallbackWrapper&&) = delete;
        ~GLFWCallbackWrapper() = delete;
        static void framebufferSizeCallbackWrapper(GLFWwindow* window, int width, int height);
        static void scrollCallbackWrapper(GLFWwindow* window, double xoffset, double yoffset);
        static void cursorPositionCallbackWrapper(GLFWwindow* window, double xpos, double ypos);
        static void mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods);
        static void setApp(App* app);
    };
public:
    int _width, _height;
    bool _hideUi;
    
    Window _window;
    Camera _camera;
    PanelManager _panelManager;// = PanelManager(this->getWindow());
    //MeshManager _meshManager;
    //Light _light;
    void start();
    void registerCallbacks();
    void initialize(int width, int height, bool hideUi);
    void processInput();
    void checkError();
    void coreRender();

    App();
    virtual ~App();

    int getWidth(){ return _width; }
    int getHeight(){ return _height; }

    struct IO;
    struct RenderVariable;
    std::unique_ptr<App::IO> _io;
    std::unique_ptr<App::RenderVariable> _renderVariable;
    Camera& getCamera() { return _camera; }
    GLFWwindow* getWindow() { return _window.getGlfwWindow();}

    virtual void setUp() {} // 처음에 사용
    virtual void preRender() {} // 루프 안에서 사용됨. 렌더 전에 사용
    virtual void render() {} // overrideable 실제 렌더링
    virtual void postRender() {} // 렌더링 이후 마무리

    virtual void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    virtual void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    virtual void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    virtual void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    GLuint addShape(std::unique_ptr<All> infos);
    std::list<std::unique_ptr<All>> _shapeLists;
    // _bufferLists // container for called in rendering loop;
    std::list<std::unique_ptr<ShapeGlBuffer>> _bufferLists;
};

#endif
#include "app.hpp"
#include "ui/base_panel.hpp"
struct App::IO
{
    bool isMouseMiddleClicked = false;
    bool isMouseLeftClicked = false;
    bool isMouseRightClicked = false;
    double mouseX = 0.0;
    double mouseY = 0.0;
    double prevMouseX = 0.0;
    double prevMouseY = 0.0;
    double deltaMouseX = 0.0;
    double deltaMouseY = 0.0;
};

struct App::RenderVariable
{
    float deltaTime = 0.0f;
    float lastFrameTime = 0.0f;
};


App::App(int width, int height, bool hideUi): 
    _width(width), _height(height), _hideUi(hideUi), _window(), _camera(), _io(new App::IO),
    _renderVariable(new App::RenderVariable)
{   
    _window = Window(_width, _height);
    BasePanel basePanel = BasePanel();
    _mainPanel.addPanel(&basePanel);
    // GL function
    glEnable(GL_DEPTH_TEST);
}

App::~App()
{

}

void App::start()
{
    setUp();
    GLFWwindow* window = _window.getGlfwWindow();
    while(!glfwWindowShouldClose(window)){
        float currentFrame = static_cast<float>(glfwGetTime());
        _renderVariable->deltaTime = currentFrame - _renderVariable->lastFrameTime;
        _renderVariable->lastFrameTime = currentFrame;
        //processInput(window, &camera);
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        _mainPanel.preRender();
        this->preRender();

        _mainPanel.render();
        this->render();
        
        _mainPanel.postRender();
        this->postRender();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

void App::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, _width, _height);
}

void App::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    float fov = _camera.getFoV();
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
        glm::vec3 cameraPos = _camera.getCameraPos();
        glm::vec3 cameraFront = _camera.getCameraLookDir();
        float cameraSpeed = static_cast<float>(10.0 * this->_renderVariable->deltaTime);
        cameraPos -= cameraSpeed * cameraFront;
        _camera.setCameraPos(cameraPos);
    if (fov > 60.0f)
        fov = 60.0f; 
    _camera.setFoV(fov);
}

void App::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    _io->mouseX = xpos;
    _io->mouseY = ypos;
    if (_io->isMouseLeftClicked == true || _io->isMouseMiddleClicked == true)
    {
        _io->deltaMouseX = _io->mouseX - _io->prevMouseX;
        _io->deltaMouseY = _io->mouseY - _io->prevMouseY;
    }
    else
    {
        _io->deltaMouseX = 0;
        _io->deltaMouseY = 0;
    }
    _io->prevMouseX = xpos;
    _io->prevMouseY = ypos;
}

void App::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    _io->deltaMouseX = 0;
    _io->deltaMouseY = 0;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        _io->isMouseLeftClicked = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        _io->isMouseLeftClicked = false;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
        _io->isMouseMiddleClicked = true;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE)
        _io->isMouseMiddleClicked = false;
}
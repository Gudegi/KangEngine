#include "window.hpp"
#include "app/app.hpp"

Window::Window(int width, int height): _width(width), _height(height){
    this->initGlfw();
    this->initGlad();
}

Window::Window(){
    _width = 1920;
    _height = 1080;
    this->initGlfw();
    this->initGlad();
}

Window::~Window()
{

}

void Window::initGlfw(){
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    _window = glfwCreateWindow(_width, _height, "KangEngine", nullptr, nullptr);
    if(_window ==NULL){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(_window);
    glfwSetWindowUserPointer(_window, this);
    glfwSetFramebufferSizeCallback(_window, this->framebufferSizeCallbackWrapper);
    glfwSetScrollCallback(_window, this->scrollCallbackWrapper);
    glfwSetCursorPosCallback(_window, this->cursorPositionCallbackWrapper);
    glfwSetMouseButtonCallback(_window, this->mouseButtonCallbackWrapper);
    glfwSwapInterval(1);
}

void Window::initGlad(){
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
    }
}

void Window::framebufferSizeCallbackWrapper(GLFWwindow* window, int width, int height)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->framebufferSizeCallback(window, width, height);
    }
}

void Window::scrollCallbackWrapper(GLFWwindow* window, double xoffset, double yoffset)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->scrollCallback(window, xoffset, yoffset);
    }
}

void Window::cursorPositionCallbackWrapper(GLFWwindow* window, double xpos, double ypos)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->cursorPositionCallback(window, xpos, ypos);
    }
}

void Window::mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->mouseButtonCallback(window, button, action, mods);
    }
}
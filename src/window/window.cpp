#include "window.hpp"
#include "app/app.hpp"

Window::Window(){
    _width = 1920;
    _height = 1080;
    this->initGlfw();
    this->initGlad();
}

Window::~Window()
{

}

void Window::init(int width, int height)
{   
    _width = width;
    _height = height;
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) {
        monitor = glfwGetPrimaryMonitor();
    }
    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    int windowWidth = static_cast<int>(_width / xscale);
    int windowHeight = static_cast<int>(_height / yscale);
    glfwSetWindowSize(_window, windowWidth, windowHeight);
    glViewport(0, 0, _width, _height);
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
    //glfwSetWindowUserPointer(_window, this);
    glfwSwapInterval(1);
}

void Window::initGlad(){
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
    }
}

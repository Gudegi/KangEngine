#include "app.hpp"

App::App(int width, int height, bool hideUi): _width(width), _height(height), _hideUi(hideUi)
{
    this->initGlfw();
    this->initGlad();

    glEnable(GL_DEPTH_TEST);

}

GLFWwindow* App::initGlfw(){
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    GLFWwindow* window = glfwCreateWindow(_width, _height, "KangEngine", nullptr, nullptr);
    if(window ==NULL){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return NULL;
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, this->framebufferSizeCallbackWrapper);
    glfwSetScrollCallback(window, this->scrollCallbackWrapper);
    glfwSetCursorPosCallback(window, this->cursorPositionCallbackWrapper);
    glfwSetMouseButtonCallback(window, this->mouseButtonCallbackWrapper);
    glfwSwapInterval(1);
    return window;
}

void App::initGlad(){
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
    }
}

void App::framebufferSizeCallbackWrapper(GLFWwindow* window, int width, int height)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->framebufferSizeCallback(window, width, height);
    }
}

void App::scrollCallbackWrapper(GLFWwindow* window, double xoffset, double yoffset)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->scrollCallback(window, xoffset, yoffset);
    }
}

void App::cursorPositionCallbackWrapper(GLFWwindow* window, double xpos, double ypos)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->cursorPositionCallback(window, xpos, ypos);
    }
}

void App::mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods)
{
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app)
    {
        app->mouseButtonCallback(window, button, action, mods);
    }
}

void App::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, _width, _height);
}

void App::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    //float fov = _camera.getFoV();
    //fov -= (float)yoffset;
    //if (fov < 1.0f)
    //    fov = 1.0f;
    //    glm::vec3 cameraPos = _camera.getCameraPos();
    //    glm::vec3 cameraFront = _camera.getCameraLookDir();
    //    float cameraSpeed = static_cast<float>(10.0 * deltaTime);
    //    cameraPos -= cameraSpeed * cameraFront;
    //    _camera.setCameraPos(cameraPos);
    //if (fov > 60.0f)
    //    fov = 60.0f; 
    //_camera.setFoV(fov);
}

void App::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    /*
    mouseX = xpos;
    mouseY = ypos;
    if (isCameraRotate == true || isCameraTranlate == true)
    {
        deltaMouseX = mouseX - prevMouseX;
        deltaMouseY = mouseY - prevMouseY;
    }
    else
    {
        deltaMouseX = 0;
        deltaMouseY = 0;
    }
    prevMouseX = xpos;
    prevMouseY = ypos;
    */
}

void App::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    /*deltaMouseX = 0;
    deltaMouseY = 0;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        isCameraRotate = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        isCameraRotate = false;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
        isCameraTranlate = true;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE)
        isCameraTranlate = false;    */
}
#include "app.hpp"
#include "ui/base_panel.hpp"
#include "utils/glm_utils.hpp"

void App::GLFWCallbackWrapper::framebufferSizeCallbackWrapper(GLFWwindow* window, int width, int height)
{
    _app->framebufferSizeCallback(window, width, height);
}

void App::GLFWCallbackWrapper::scrollCallbackWrapper(GLFWwindow* window, double xoffset, double yoffset)
{
    _app->scrollCallback(window, xoffset, yoffset);
}

void App::GLFWCallbackWrapper::cursorPositionCallbackWrapper(GLFWwindow* window, double xpos, double ypos)
{
    _app->cursorPositionCallback(window, xpos, ypos);
}

void App::GLFWCallbackWrapper::mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods)
{
    _app->mouseButtonCallback(window, button, action, mods);
}

void App::GLFWCallbackWrapper::setApp(App* app)
{
    GLFWCallbackWrapper::_app = app;
}

App* App::GLFWCallbackWrapper::_app = nullptr;

void App::registerCallbacks()
{
    GLFWCallbackWrapper::setApp(this);
    GLFWwindow* window = _window.getGlfwWindow();
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, GLFWCallbackWrapper::framebufferSizeCallbackWrapper);
    glfwSetScrollCallback(window, GLFWCallbackWrapper::scrollCallbackWrapper);
    glfwSetCursorPosCallback(window, GLFWCallbackWrapper::cursorPositionCallbackWrapper);
    glfwSetMouseButtonCallback(window, GLFWCallbackWrapper::mouseButtonCallbackWrapper);
}
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


App::App(): 
    _width(), _height(),
    _hideUi(), _window(), _camera(), 
    _io(new App::IO),
    _renderVariable(new App::RenderVariable)
{   

}

App::~App()
{

}

void App::initialize(int width, int height, bool hideUi)
{
    _width = width;
    _height = height;
    _hideUi = hideUi;
    _window.init(_width, _height);
    registerCallbacks();
    glm::vec3 cameraPos = glm::vec3(3, 2, -1);
    glm::vec3 cameraTarget = glm::vec3(0, 0, 0);
    _camera.init(cameraPos, cameraTarget, 'y');
    _camera.updateProjMatrix(_width, _height);
    _panelManager.init(this->getWindow());
    //BasePanel basePanel = BasePanel();
    //_panelManager.addPanel(&basePanel);
    _panelManager.addPanel(std::make_unique<BasePanel>());
    // GL function
    glEnable(GL_DEPTH_TEST);
}

void App::start()
{
    setUp();
    GLFWwindow* window = _window.getGlfwWindow();
    while(!glfwWindowShouldClose(window)){
        float currentFrame = static_cast<float>(glfwGetTime());
        _renderVariable->deltaTime = currentFrame - _renderVariable->lastFrameTime;
        _renderVariable->lastFrameTime = currentFrame;
        processInput();
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        _panelManager.preRender();
        this->preRender();
        coreRender();
        _panelManager.render();
        this->render();
        _panelManager.postRender();
        this->postRender();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

void App::coreRender()
{
    //int start_instance_idx = 0;
    int a = 0;
    for (const auto &buffer : _bufferLists) 
    {   
        buffer->vao->bind();
        glDrawElements(GL_TRIANGLES, buffer->numIndices, GL_UNSIGNED_INT, 0);
        checkError();
        //start_instance_idx += num_instances;
        VAO::vaoUnBind();
        checkError();
        a += 1;
    }
}

void App::checkError()
{
    GLenum err;
    if ((err = glGetError()) != GL_NO_ERROR)
    {
        std::string errStr = "";
        switch (err)
        {
            case GL_INVALID_ENUM:      errStr = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE:     errStr = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: errStr = "GL_INVALID_OPERATION"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: errStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
            case GL_OUT_OF_MEMORY:   errStr = "GL_OUT_OF_MEMORY"; break;
            case GL_STACK_UNDERFLOW: errStr = "GL_STACK_UNDERFLOW"; break;
            case GL_STACK_OVERFLOW:  errStr = "GL_STACK_OVERFLOW"; break;
        }
        std::cout << errStr << std::endl;
    }
}

GLuint App::addShape(std::unique_ptr<All> infos)
{
    auto bufferInfos = std::make_unique<ShapeGlBuffer>();
    bufferInfos->vao = new VAO;
    bufferInfos->vao->bind();
    bufferInfos->vbo = new VBO;
    bufferInfos->vbo->bind();
    bufferInfos->ebo = new EBO;
    bufferInfos->ebo->bind();
    bufferInfos->numIndices = infos->indices.size();
    
    bufferInfos->vbo->setData(sizeof(VertexAttrib) * infos->vertexAttrib.size(), &infos->vertexAttrib[0], GL_STATIC_DRAW); // the size of vector type * the number of vectors
    bufferInfos->ebo->setData(sizeof(unsigned int) * infos->indices.size(), &infos->indices[0], GL_STATIC_DRAW);
    
    bufferInfos->vao->setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)0);
    bufferInfos->vao->setVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)offsetof(VertexAttrib, normal));
    bufferInfos->vao->setVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)offsetof(VertexAttrib, uv));
    bufferInfos->vao->unBind();//VAO::vaoUnBind();
    checkError();

    /*
    //VAO vao;
    //vao.bind();
    //VBO vbo;
    //vbo.bind();
    //EBO ebo;
    //ebo.bind();

    //vbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
    vbo.setData(sizeof(infos->vertexAttrib), &infos->vertexAttrib[0], GL_STATIC_DRAW);
    ebo.setData(sizeof(infos->indices), &infos->indices[0], GL_STATIC_DRAW);
    //ebo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
    vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)0);
    // TODO: fixme 
    vao.setVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)offsetof(VertexAttrib, normal));
    vao.setVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)offsetof(VertexAttrib, uv));
    //vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    //vao.setVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
    vao.unBind();//VAO::vaoUnBind();
    */
    VAO::vaoUnBind();    
    _shapeLists.push_back(std::move(infos));
    _bufferLists.push_back(std::move(bufferInfos));

    GLuint shapeIdx = _shapeLists.size() - 1;
    return shapeIdx;

}

//////////////// Call backs ////////////////////////////////////////////////////////
void App::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    _width = width;
    _height = height;
    glViewport(0, 0, _width, _height);
    _camera.updateProjMatrix(_width, _height);
}

void App::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    float fov = _camera.getFoV();
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
        glm::vec3 cameraPos = _camera.getCameraPos();
        glm::vec3 cameraFront = _camera.getCameraLookDir();
        float cameraSpeed = static_cast<float>(10.0 * _renderVariable->deltaTime);
        cameraPos -= cameraSpeed * cameraFront;
        _camera.setCameraPos(cameraPos);
    if (fov > 60.0f)
        fov = 60.0f; 
    _camera.setFoV(fov);
    _camera.updateProjMatrix(_width, _height);
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

void App::processInput() 
    {
        GLFWwindow* window = this->getWindow();
        float cameraSpeed = static_cast<float>(15.0 * _renderVariable->deltaTime);
        float scaleDistance = _camera.getCamToLookDistance();
        glm::vec3 cameraPos = _camera.getCameraPos();
        glm::vec3 cameraFront = _camera.getCameraLookDir();
        glm::vec3 cameraUp = _camera.getCameraUpDir();
        glm::vec3 cameraRight = _camera.getCameraRightDir();
        glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);
    
        if( glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        
        // look at specifc target or // look at forward direction 
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            {
                if (_io->isMouseLeftClicked == true)
                {   
                    glm::vec3 nextCameraPos = cameraPos + cameraSpeed * cameraUp;
                    glm::vec3 nextCameraFront = _camera.getTargetPos() - nextCameraPos; // No normalize because it is just for sign checking.
                    if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                        return; // To prevent flipping, compare nextCameraFront with cameraFront.
                    cameraPos = nextCameraPos;
                }
                else
                {
                    glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                    _camera.setTargetPos(cameraPos + scaledFront);
                    //cameraPos += cameraSpeed * cameraFront;
                    cameraPos += cameraSpeed * cameraUp;
                    _camera.setTargetPos(cameraPos + scaledFront); // prevent disconnect
                }
                _camera.setCameraPos(cameraPos);
            }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            {        
                if (_io->isMouseLeftClicked == true)
                {
                    glm::vec3 nextCameraPos = cameraPos - cameraSpeed * cameraUp;
                    glm::vec3 nextCameraFront = _camera.getTargetPos() - nextCameraPos;
                    if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                        return;
                    cameraPos = nextCameraPos;
                }
                else
                {
                    glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                    _camera.setTargetPos(cameraPos + scaledFront);
                    //cameraPos -= cameraSpeed * cameraFront;
                    cameraPos -= cameraSpeed * cameraUp;
                    _camera.setTargetPos(cameraPos + scaledFront);
                }
                _camera.setCameraPos(cameraPos);
            }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            {
                if (_io->isMouseLeftClicked == true)
                {
                    cameraPos -= cameraSpeed * cameraRight;
                }
                else
                {
                    glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                    _camera.setTargetPos(cameraPos + scaledFront);
                    cameraPos -= cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
                    _camera.setTargetPos(cameraPos + scaledFront);
                }
                _camera.setCameraPos(cameraPos);
            }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            {
                if (_io->isMouseLeftClicked == true)
                {
                    cameraPos += cameraSpeed * cameraRight;
                }
                else
                {
                    glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                    _camera.setTargetPos(cameraPos + scaledFront);
                    cameraPos += cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
                    _camera.setTargetPos(cameraPos + scaledFront);
                }
                _camera.setCameraPos(cameraPos);
            }
    
        if (_io->isMouseLeftClicked) 
        {
            const double DRAG_THRESHOLD = 2.0;
            if (sqrt(_io->deltaMouseX * _io->deltaMouseX + _io->deltaMouseY * _io->deltaMouseY) < DRAG_THRESHOLD)
            {
                return;
            }
            if (_io->deltaMouseX > 0) 
            {
                cameraPos -= cameraSpeed * cameraRight;
            }
            if (_io->deltaMouseX < 0)
            { 
                cameraPos += cameraSpeed * cameraRight;
            }
            if (_io->deltaMouseY > 0) 
            {
                glm::vec3 nextCameraPos = cameraPos - cameraSpeed * cameraUp;
                glm::vec3 nextCameraFront = _camera.getTargetPos() - nextCameraPos;
                if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                    return;
                cameraPos = nextCameraPos;
            }
            if (_io->deltaMouseY < 0) 
            {
                glm::vec3 nextCameraPos = cameraPos + cameraSpeed * cameraUp;
                glm::vec3 nextCameraFront = _camera.getTargetPos() - nextCameraPos; // No normalize because it is just for sign checking.
                if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                    return; // To prevent flipping, compare nextCameraFront with cameraFront.
                cameraPos = nextCameraPos;
            }
            _camera.setCameraPos(cameraPos);
        }
    
        if (_io->isMouseMiddleClicked) 
        {
            const double DRAG_THRESHOLD = 2.0;
            if (sqrt(_io->deltaMouseX * _io->deltaMouseX + _io->deltaMouseY * _io->deltaMouseY) < DRAG_THRESHOLD)
            {
                return;
            }
            if (_io->deltaMouseX > 0) 
            {
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                _camera.setTargetPos(cameraPos + scaledFront);
                cameraPos -= cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
                _camera.setTargetPos(cameraPos + scaledFront);
            }
            if (_io->deltaMouseX < 0)
            { 
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                _camera.setTargetPos(cameraPos + scaledFront);
                cameraPos += cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
                _camera.setTargetPos(cameraPos + scaledFront);
            }
            if (_io->deltaMouseY > 0) 
            {
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                _camera.setTargetPos(cameraPos + scaledFront);
                //cameraPos += cameraSpeed * cameraFront;
                cameraPos += cameraSpeed * cameraUp;
                _camera.setTargetPos(cameraPos + scaledFront);
            }
            if (_io->deltaMouseY < 0) 
            {
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                _camera.setTargetPos(cameraPos + scaledFront);
                //cameraPos -= cameraSpeed * cameraFront;
                cameraPos -= cameraSpeed * cameraUp;
                _camera.setTargetPos(cameraPos + scaledFront);
            }
            _camera.setCameraPos(cameraPos);
        }
    }

//////////////// Call backs ////////////////////////////////////////////////////////
#include "app.hpp"
#include "ui/base_panel.hpp"
#include "utils/glm_utils.hpp"

namespace KE {

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
    _renderWireframe(false),
    _io(new App::IO),
    _renderVariable(new App::RenderVariable)
{   

}

App::~App()
{

}

void App::initialize(int width, int height, bool hideUi, Backend::BackendType backendType)
{
    _width = width;
    _height = height;
    _hideUi = hideUi;

    // Initialize graphics device first
    _graphicsDevice = Backend::GraphicsFactory::createDevice(backendType);
    if (!_graphicsDevice) {
        std::cerr << "Failed to create graphics device" << std::endl;
        return;
    }

    _window.init(_width, _height);
    registerCallbacks();

    // Initialize graphics device after OpenGL context is created
    _graphicsDevice->initialize();

    glm::vec3 cameraPos = glm::vec3(3, 2, -1);
    glm::vec3 cameraTarget = glm::vec3(0, 0, 0);
    _camera.init(cameraPos, cameraTarget, 'y');
    _camera.updateProjMatrix(_width, _height);
    _panelManager.init(this->getWindow());
    _panelManager.addPanel(std::make_unique<BasePanel>());

    // Use backend abstraction instead of direct OpenGL
    _graphicsDevice->setDepthTest(true);
}

// for entire process in c++ //////////////////////////////////////////////////////////////////////
void App::start()
{
    setup();
    GLFWwindow* window = _window.getGlfwWindow();
    while(!glfwWindowShouldClose(window)){
        float currentFrame = static_cast<float>(glfwGetTime());
        _renderVariable->deltaTime = currentFrame - _renderVariable->lastFrameTime;
        _renderVariable->lastFrameTime = currentFrame;
        processInput();
        _viewMatrix = _camera.getViewMatrix();
        _projectionMatrix = _camera.getProjMatrix();
        
        _graphicsDevice->setClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        _graphicsDevice->clear(0.2f, 0.3f, 0.3f, 1.0f);
        _panelManager.preRender();
        this->preRender();
        _panelManager.render();
        coreRender();
        this->render();
        _panelManager.postRender();
        this->postRender();
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Fill);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

void App::coreRender()
{
    ImGui::Checkbox("Wireframe", &_renderWireframe);

    if (_renderWireframe == true)
    {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Line);
    }
    else
    {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Fill);
    }

    for (const auto &buffer : _bufferLists)
    {
        // Backend::Shader와 KE::Shader 둘 다 지원
        if (buffer->backendShader) {
            // Backend::Shader 사용
            buffer->backendShader->use();
            buffer->backendShader->setMat4("view", _viewMatrix);
            buffer->backendShader->setMat4("projection", _projectionMatrix);
        } else if (buffer->shader) {
            // KE::Shader 사용 (기존 방식)
            buffer->shader->use();
            buffer->shader->setMat4("view", _viewMatrix);
            buffer->shader->setMat4("projection", _projectionMatrix);
        }

        buffer->vertexArray->bind();
        _graphicsDevice->drawIndexed(buffer->numIndices);
        checkError();
        buffer->vertexArray->unbind();
        checkError();
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////


/// for each step rendering for both c++ and python ////////////////////////////////////////////
void App::draw()
{
    //TODO : update me
    /*
    float currentFrame = static_cast<float>(glfwGetTime());
    _renderVariable->deltaTime = currentFrame - _renderVariable->lastFrameTime;
    _renderVariable->lastFrameTime = currentFrame;
    processInput();
    _viewMatrix = _camera.getViewMatrix();
    _projectionMatrix = _camera.getProjMatrix();
    
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    */
}
////////////////////////////////////////////////////////////////////////////////////////////////

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

size_t App::addShape(Shader* shader, std::unique_ptr<All> infos)
{
    if (!infos || infos->vertexAttrib.empty() || infos->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        auto bufferInfos = std::make_unique<ShapeRenderBuffer>();
        bufferInfos->shader = shader;

        if (shader) {
            shader->use();
        }

        // Create buffers using backend abstraction
        bufferInfos->vertexBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Vertex,
            sizeof(VertexAttrib) * infos->vertexAttrib.size(),
            infos->vertexAttrib.data()
        );

        bufferInfos->indexBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Index,
            sizeof(unsigned int) * infos->indices.size(),
            infos->indices.data()
        );

        // Create and configure vertex array
        bufferInfos->vertexArray = _graphicsDevice->createVertexArray();
        bufferInfos->vertexArray->bind();

        // Set vertex and index buffers
        bufferInfos->vertexArray->setVertexBuffer(bufferInfos->vertexBuffer.get());
        bufferInfos->vertexArray->setIndexBuffer(bufferInfos->indexBuffer.get());

        // Define vertex attributes
        constexpr int POSITION_ATTRIB = 0;
        constexpr int NORMAL_ATTRIB = 1;
        constexpr int UV_ATTRIB = 2;
        constexpr int POSITION_SIZE = 3;
        constexpr int NORMAL_SIZE = 3;
        constexpr int UV_SIZE = 2;

        Backend::VertexAttribute positionAttr = {
            POSITION_ATTRIB, POSITION_SIZE, Backend::VertexAttributeType::Float,
            false, sizeof(VertexAttrib), 0
        };
        bufferInfos->vertexArray->setVertexAttribute(positionAttr);

        Backend::VertexAttribute normalAttr = {
            NORMAL_ATTRIB, NORMAL_SIZE, Backend::VertexAttributeType::Float,
            false, sizeof(VertexAttrib), offsetof(VertexAttrib, normal)
        };
        bufferInfos->vertexArray->setVertexAttribute(normalAttr);

        Backend::VertexAttribute uvAttr = {
            UV_ATTRIB, UV_SIZE, Backend::VertexAttributeType::Float,
            false, sizeof(VertexAttrib), offsetof(VertexAttrib, uv)
        };
        bufferInfos->vertexArray->setVertexAttribute(uvAttr);

        bufferInfos->vertexArray->unbind();
        bufferInfos->numIndices = infos->indices.size();
        checkError();

        _shapeLists.push_back(std::move(infos));
        _bufferLists.push_back(std::move(bufferInfos));

        return _shapeLists.size() - 1;

    }
    catch (const std::exception& e) {
        std::cerr << "Failed to add shape: " << e.what() << std::endl;
        return static_cast<size_t>(-1);
    }
}

size_t App::addShape(Backend::Shader* shader, std::unique_ptr<All> infos)
{
    if (!infos || infos->vertexAttrib.empty() || infos->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        auto bufferInfos = std::make_unique<ShapeRenderBuffer>();
        bufferInfos->backendShader = shader;  // Backend::Shader 사용

        if (shader) {
            shader->use();
        }

        // Create buffers using backend abstraction
        bufferInfos->vertexBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Vertex,
            sizeof(VertexAttrib) * infos->vertexAttrib.size(),
            infos->vertexAttrib.data()
        );

        bufferInfos->indexBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Index,
            sizeof(unsigned int) * infos->indices.size(),
            infos->indices.data()
        );

        // Create and configure vertex array
        bufferInfos->vertexArray = _graphicsDevice->createVertexArray();
        bufferInfos->vertexArray->bind();

        // Set vertex and index buffers
        bufferInfos->vertexArray->setVertexBuffer(bufferInfos->vertexBuffer.get());
        bufferInfos->vertexArray->setIndexBuffer(bufferInfos->indexBuffer.get());

        // Define vertex attributes
        constexpr int POSITION_ATTRIB = 0;
        constexpr int NORMAL_ATTRIB = 1;
        constexpr int UV_ATTRIB = 2;
        constexpr int POSITION_SIZE = 3;
        constexpr int NORMAL_SIZE = 3;
        constexpr int UV_SIZE = 2;

        Backend::VertexAttribute positionAttr = {
            POSITION_ATTRIB, POSITION_SIZE, Backend::VertexAttributeType::Float,
            false, sizeof(VertexAttrib), 0
        };
        bufferInfos->vertexArray->setVertexAttribute(positionAttr);

        Backend::VertexAttribute normalAttr = {
            NORMAL_ATTRIB, NORMAL_SIZE, Backend::VertexAttributeType::Float,
            false, sizeof(VertexAttrib), offsetof(VertexAttrib, normal)
        };
        bufferInfos->vertexArray->setVertexAttribute(normalAttr);

        Backend::VertexAttribute uvAttr = {
            UV_ATTRIB, UV_SIZE, Backend::VertexAttributeType::Float,
            false, sizeof(VertexAttrib), offsetof(VertexAttrib, uv)
        };
        bufferInfos->vertexArray->setVertexAttribute(uvAttr);

        bufferInfos->vertexArray->unbind();
        bufferInfos->numIndices = infos->indices.size();
        checkError();

        _shapeLists.push_back(std::move(infos));
        _bufferLists.push_back(std::move(bufferInfos));

        return _shapeLists.size() - 1;

    }
    catch (const std::exception& e) {
        std::cerr << "Failed to add shape with Backend::Shader: " << e.what() << std::endl;
        return static_cast<size_t>(-1);
    }
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

} // namespace KE

//////////////// Call backs ////////////////////////////////////////////////////////
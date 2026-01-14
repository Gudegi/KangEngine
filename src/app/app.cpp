#include "app.hpp"
#include "ui/base_panel.hpp"
#include "utils/glm_utils.hpp"
#include "utils/print_debug.hpp"

namespace KE {

void App::GLFWCallbackWrapper::framebufferSizeCallbackWrapper(
    GLFWwindow* window, int width, int height) {
    _app->framebufferSizeCallback(window, width, height);
}

void App::GLFWCallbackWrapper::scrollCallbackWrapper(GLFWwindow* window,
                                                     double xoffset,
                                                     double yoffset) {
    _app->scrollCallback(window, xoffset, yoffset);
}

void App::GLFWCallbackWrapper::cursorPositionCallbackWrapper(GLFWwindow* window,
                                                             double xpos,
                                                             double ypos) {
    _app->cursorPositionCallback(window, xpos, ypos);
}

void App::GLFWCallbackWrapper::mouseButtonCallbackWrapper(GLFWwindow* window,
                                                          int button,
                                                          int action,
                                                          int mods) {
    _app->mouseButtonCallback(window, button, action, mods);
}

void App::GLFWCallbackWrapper::setApp(App* app) {
    GLFWCallbackWrapper::_app = app;
}

App* App::GLFWCallbackWrapper::_app = nullptr;

void App::registerCallbacks() {
    GLFWCallbackWrapper::setApp(this);
    GLFWwindow* window = _window.getGlfwWindow();
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(
        window, GLFWCallbackWrapper::framebufferSizeCallbackWrapper);
    glfwSetScrollCallback(window, GLFWCallbackWrapper::scrollCallbackWrapper);
    glfwSetCursorPosCallback(
        window, GLFWCallbackWrapper::cursorPositionCallbackWrapper);
    glfwSetMouseButtonCallback(window,
                               GLFWCallbackWrapper::mouseButtonCallbackWrapper);
}
struct App::IO {
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

struct App::RenderVariable {
    float deltaTime = 0.0f;
    float lastFrameTime = 0.0f;
};

App::App()
    : _width(), _height(), _hideUi(), _window(), _camera(),
      _renderWireframe(false), _io(new App::IO),
      _renderVariable(new App::RenderVariable) {}

App::~App() {}

void App::initialize(int width, int height, bool hideUi,
                     Backend::BackendType graphicsBackendType,
                     Scene::BackendType sceneBackendType) {
    _width = width;
    _height = height;
    _hideUi = hideUi;

    // Initialize graphics device first
    _graphicsDevice =
        Backend::GraphicsFactory::createDevice(graphicsBackendType);
    if (!_graphicsDevice) {
        std::cerr << "Failed to create graphics device" << std::endl;
        return;
    }
    _scene = Scene::SceneFactory::createBackend(sceneBackendType);
    if (!_scene) {
        std::cerr << "Failed to create scene" << std::endl;
        return;
    }

    _window.init(_width, _height);
    registerCallbacks();

    // Initialize graphics device after OpenGL context is created
    _graphicsDevice->initialize();

    glm::vec3 cameraPos = glm::vec3(30, 15, 0);
    glm::vec3 cameraTarget = glm::vec3(0, 0, 0);
    _camera.init(cameraPos, cameraTarget, 'y');
    _camera.updateProjMatrix(_width, _height);
    _panelManager.init(this->getWindow());
    _panelManager.addPanel(std::make_unique<BasePanel>());

    // Use backend abstraction instead of direct OpenGL
    _graphicsDevice->setDepthTest(true);
}

// for entire process in c++
// //////////////////////////////////////////////////////////////////////
void App::start() {
    setup();
    GLFWwindow* window = _window.getGlfwWindow();
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        _renderVariable->deltaTime =
            currentFrame - _renderVariable->lastFrameTime;
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

void App::coreRender() {
    ImGui::Checkbox("Wireframe", &_renderWireframe);

    if (_renderWireframe == true) {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Line);
    } else {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Fill);
    }

    if (_scene) {
        const auto& sceneBuffers = _scene->getBufferLists();
        for (const auto& buffer : sceneBuffers) {
            if (buffer && buffer->backendShader) {
                buffer->backendShader->use();
                buffer->backendShader->setMat4("view", _viewMatrix);
                buffer->backendShader->setMat4("projection", _projectionMatrix);

                buffer->vertexArray->bind();
                _graphicsDevice->drawIndexed(buffer->numIndices);
                buffer->vertexArray->unbind();
                checkError();
            }
        }
    } else {
        for (const auto& buffer : _bufferLists) {
            if (buffer->backendShader) {
                // Backend::Shader method
                buffer->backendShader->use();
                buffer->backendShader->setMat4("view", _viewMatrix);
                buffer->backendShader->setMat4("projection", _projectionMatrix);
            } else {
                throw std::runtime_error("Unknown render type");
            }

            buffer->vertexArray->bind();
            _graphicsDevice->drawIndexed(buffer->numIndices);
            checkError();
            buffer->vertexArray->unbind();
            checkError();
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////

/// for each step rendering for both c++ and python
/// ////////////////////////////////////////////
void App::draw() {
    // TODO : update me
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

void App::checkError() { _graphicsDevice->checkError(); }

size_t App::addShape(Backend::Shader* shader,
                     std::shared_ptr<Scene::MeshData> meshData) {
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        auto bufferInfos = std::make_shared<Scene::ShapeRenderBuffer>();
        bufferInfos->backendShader = shader;

        if (shader) {
            shader->use();
        }

        // Create separate buffers for each attribute using meshData directly
        // Position buffer (location 0)
        auto positionBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Vertex,
            sizeof(glm::vec3) * meshData->vertices.size(),
            meshData->vertices.data());

        // Normal buffer (location 1) - if exists
        std::unique_ptr<Backend::Buffer> normalBuffer;
        if (!meshData->normals.empty()) {
            normalBuffer = _graphicsDevice->createBuffer(
                Backend::BufferType::Vertex,
                sizeof(glm::vec3) * meshData->normals.size(),
                meshData->normals.data());
        }

        // UV buffer (location 2) - if exists
        std::unique_ptr<Backend::Buffer> uvBuffer;
        if (!meshData->uvs.empty()) {
            uvBuffer = _graphicsDevice->createBuffer(
                Backend::BufferType::Vertex,
                sizeof(glm::vec2) * meshData->uvs.size(), meshData->uvs.data());
        }

        // Index buffer
        bufferInfos->indexBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Index,
            sizeof(unsigned int) * meshData->indices.size(),
            meshData->indices.data());

        // Create and configure vertex array
        bufferInfos->vertexArray = _graphicsDevice->createVertexArray();
        bufferInfos->vertexArray->bind();

        // Set index buffer
        bufferInfos->vertexArray->setIndexBuffer(
            bufferInfos->indexBuffer.get());

        // Position attribute (location 0)
        positionBuffer->bind();
        Backend::VertexAttribute positionAttr = {
            0, 3, Backend::VertexAttributeType::Float, false, sizeof(glm::vec3),
            0};
        bufferInfos->vertexArray->setVertexAttribute(positionAttr);

        // Normal attribute (location 1)
        if (normalBuffer) {
            normalBuffer->bind();
            Backend::VertexAttribute normalAttr = {
                1,
                3,
                Backend::VertexAttributeType::Float,
                false,
                sizeof(glm::vec3),
                0};
            bufferInfos->vertexArray->setVertexAttribute(normalAttr);
        }

        // UV attribute (location 2)
        if (uvBuffer) {
            uvBuffer->bind();
            Backend::VertexAttribute uvAttr = {
                2,
                2,
                Backend::VertexAttributeType::Float,
                false,
                sizeof(glm::vec2),
                0};
            bufferInfos->vertexArray->setVertexAttribute(uvAttr);
        }

        bufferInfos->vertexArray->unbind();
        bufferInfos->numIndices = meshData->indices.size();

        // Store position buffer (we need to keep at least one buffer alive)
        bufferInfos->vertexBuffer = std::move(positionBuffer);
        // Note: normalBuffer and uvBuffer will be destroyed here,
        // but OpenGL has already copied the data

        checkError();

        //_bufferLists.push_back(std::move(bufferInfos));
        _scene->addRenderable(bufferInfos);
        return _scene->getBufferLists().size() - 1;

    } catch (const std::exception& e) {
        std::cerr << "Failed to add shape with Scene::MeshData: " << e.what()
                  << std::endl;
        return static_cast<size_t>(-1);
    }
}

size_t App::addShape(PhongMaterial* material,
                     std::shared_ptr<Scene::MeshData> meshData) {
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        auto bufferInfos = std::make_shared<Scene::ShapeRenderBuffer>();
        bufferInfos->backendShader =
            material->getShader(); // TODO: need to fix?

        if (bufferInfos->backendShader) {
            material->bind();
        }

        // Create separate buffers for each attribute using meshData directly
        // Position buffer (location 0)
        auto positionBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Vertex,
            sizeof(glm::vec3) * meshData->vertices.size(),
            meshData->vertices.data());

        // Normal buffer (location 1) - if exists
        std::unique_ptr<Backend::Buffer> normalBuffer;
        if (!meshData->normals.empty()) {
            normalBuffer = _graphicsDevice->createBuffer(
                Backend::BufferType::Vertex,
                sizeof(glm::vec3) * meshData->normals.size(),
                meshData->normals.data());
        }

        // UV buffer (location 2) - if exists
        std::unique_ptr<Backend::Buffer> uvBuffer;
        if (!meshData->uvs.empty()) {
            uvBuffer = _graphicsDevice->createBuffer(
                Backend::BufferType::Vertex,
                sizeof(glm::vec2) * meshData->uvs.size(), meshData->uvs.data());
        }

        // Index buffer
        bufferInfos->indexBuffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Index,
            sizeof(unsigned int) * meshData->indices.size(),
            meshData->indices.data());

        // Create and configure vertex array
        bufferInfos->vertexArray = _graphicsDevice->createVertexArray();
        bufferInfos->vertexArray->bind();

        // Set index buffer
        bufferInfos->vertexArray->setIndexBuffer(
            bufferInfos->indexBuffer.get());

        // Position attribute (location 0)
        positionBuffer->bind();
        Backend::VertexAttribute positionAttr = {
            0, 3, Backend::VertexAttributeType::Float, false, sizeof(glm::vec3),
            0};
        bufferInfos->vertexArray->setVertexAttribute(positionAttr);

        // Normal attribute (location 1)
        if (normalBuffer) {
            normalBuffer->bind();
            Backend::VertexAttribute normalAttr = {
                1,
                3,
                Backend::VertexAttributeType::Float,
                false,
                sizeof(glm::vec3),
                0};
            bufferInfos->vertexArray->setVertexAttribute(normalAttr);
        }

        // UV attribute (location 2)
        if (uvBuffer) {
            uvBuffer->bind();
            Backend::VertexAttribute uvAttr = {
                2,
                2,
                Backend::VertexAttributeType::Float,
                false,
                sizeof(glm::vec2),
                0};
            bufferInfos->vertexArray->setVertexAttribute(uvAttr);
        }

        bufferInfos->vertexArray->unbind();
        bufferInfos->numIndices = meshData->indices.size();
        bufferInfos->vertexBuffer = std::move(positionBuffer);

        checkError();
        _scene->addRenderable(bufferInfos);
        return _scene->getBufferLists().size() - 1;
    } catch (const std::exception& e) {
        std::cerr << "Failed to add shape with PhongMaterial: " << e.what()
                  << std::endl;
        return static_cast<size_t>(-1);
    }
}

//////////////// Call backs
///////////////////////////////////////////////////////////
void App::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    _width = width;
    _height = height;
    _graphicsDevice->setViewport(0, 0, _width, _height);
    _camera.updateProjMatrix(_width, _height);
}

void App::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    float fov = _camera.getFoV();
    fov -= (float)yoffset;

    /*
    glm::vec3 cameraPos = _camera.getCameraPos();
    glm::vec3 cameraFront = _camera.getCameraLookDir();
    float cameraSpeed = static_cast<float>(10.0 * _renderVariable->deltaTime);
    cameraPos -= cameraSpeed * cameraFront;
    _camera.setCameraPos(cameraPos);
    */

    _camera.setFoV(fov);
    _camera.updateProjMatrix(_width, _height);
}

void App::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    _io->mouseX = xpos;
    _io->mouseY = ypos;
    if (_io->isMouseLeftClicked == true || _io->isMouseMiddleClicked == true ||
        _io->isMouseRightClicked == true) {
        _io->deltaMouseX = _io->mouseX - _io->prevMouseX;
        _io->deltaMouseY = _io->mouseY - _io->prevMouseY;
    } else {
        _io->deltaMouseX = 0;
        _io->deltaMouseY = 0;
    }
    _io->prevMouseX = xpos;
    _io->prevMouseY = ypos;
}

void App::mouseButtonCallback(GLFWwindow* window, int button, int action,
                              int mods) {
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
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
        _io->isMouseRightClicked = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
        _io->isMouseRightClicked = false;
}

void App::processInput() {
    // Prevent manipulations while ImGui using the mouse.
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    GLFWwindow* window = this->getWindow();
    float cameraSpeed = static_cast<float>(15.0 * _renderVariable->deltaTime);
    float scaleDistance = _camera.getCamToLookDistance();
    glm::vec3 cameraPos = _camera.getCameraPos();
    glm::vec3 cameraFront = _camera.getCameraLookDir();
    glm::vec3 cameraUp = _camera.getCameraUpDir();
    glm::vec3 cameraRight = _camera.getCameraRightDir();
    glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // look at specifc target or // look at forward direction
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        if (_io->isMouseLeftClicked == true) {
            glm::vec3 nextCameraPos = cameraPos + cameraSpeed * cameraUp;
            glm::vec3 nextCameraFront =
                _camera.getTargetPos() -
                nextCameraPos; // No normalize because it is just for sign
                               // checking.
            if (((nextCameraFront.x * cameraFront.x) < 0) &&
                (nextCameraFront.z * cameraFront.z) < 0)
                return; // To prevent flipping, compare nextCameraFront with
                        // cameraFront.
            cameraPos = nextCameraPos;
        } else {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            // cameraPos += cameraSpeed * cameraFront;
            cameraPos += cameraSpeed * cameraUp;
            _camera.setTargetPos(cameraPos + scaledFront); // prevent disconnect
        }
        _camera.setCameraPos(cameraPos);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        if (_io->isMouseLeftClicked == true) {
            glm::vec3 nextCameraPos = cameraPos - cameraSpeed * cameraUp;
            glm::vec3 nextCameraFront = _camera.getTargetPos() - nextCameraPos;
            if (((nextCameraFront.x * cameraFront.x) < 0) &&
                (nextCameraFront.z * cameraFront.z) < 0)
                return;
            cameraPos = nextCameraPos;
        } else {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            // cameraPos -= cameraSpeed * cameraFront;
            cameraPos -= cameraSpeed * cameraUp;
            _camera.setTargetPos(cameraPos + scaledFront);
        }
        _camera.setCameraPos(cameraPos);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        if (_io->isMouseLeftClicked == true) {
            cameraPos -= cameraSpeed * cameraRight;
        } else {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            cameraPos -=
                cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
            _camera.setTargetPos(cameraPos + scaledFront);
        }
        _camera.setCameraPos(cameraPos);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        if (_io->isMouseLeftClicked == true) {
            cameraPos += cameraSpeed * cameraRight;
        } else {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            cameraPos +=
                cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
            _camera.setTargetPos(cameraPos + scaledFront);
        }
        _camera.setCameraPos(cameraPos);
    }

    if (_io->isMouseLeftClicked) {
        // Orbit cam
        float dx = _io->deltaMouseX * 0.1f;
        float dy = _io->deltaMouseY * 0.1f;

        _camera.azimuth += dx;
        _camera.pole -= dy; // (mouse down -> cam up)

        const float EPSILON = 0.01f;
        _camera.pole = glm::clamp(_camera.pole, EPSILON, 180.0f - EPSILON);

        float phi = glm::radians(_camera.azimuth);
        float theta = glm::radians(_camera.pole);
        float r = scaleDistance;

        glm::vec3 targetPos = _camera.getTargetPos();
        glm::vec3 newCameraPos;

        // Y-up
        newCameraPos.x = targetPos.x + r * glm::sin(theta) * glm::cos(phi);
        newCameraPos.y = targetPos.y + r * glm::cos(theta);
        newCameraPos.z = targetPos.z + r * glm::sin(theta) * glm::sin(phi);
        _camera.setCameraPos(newCameraPos);
    }

    if (_io->isMouseRightClicked) {
        const double DRAG_THRESHOLD = 2.0;
        if (sqrt(_io->deltaMouseX * _io->deltaMouseX +
                 _io->deltaMouseY * _io->deltaMouseY) < DRAG_THRESHOLD) {
            return;
        }
        if (_io->deltaMouseX > 0) {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            cameraPos -=
                cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
            _camera.setTargetPos(cameraPos + scaledFront);
        }
        if (_io->deltaMouseX < 0) {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            cameraPos +=
                cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
            _camera.setTargetPos(cameraPos + scaledFront);
        }
        if (_io->deltaMouseY > 0) {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            // cameraPos += cameraSpeed * cameraFront;
            cameraPos += cameraSpeed * cameraUp;
            _camera.setTargetPos(cameraPos + scaledFront);
        }
        if (_io->deltaMouseY < 0) {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            _camera.setTargetPos(cameraPos + scaledFront);
            // cameraPos -= cameraSpeed * cameraFront;
            cameraPos -= cameraSpeed * cameraUp;
            _camera.setTargetPos(cameraPos + scaledFront);
        }
        /*
        // panning
        glm::vec3 rightMove = static_cast<float>(-_io->deltaMouseX) *
        cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
        glm::vec3 upMove = static_cast<float>(-_io->deltaMouseY) * cameraSpeed *
        cameraUp; glm::vec3 totalMove = rightMove + upMove;

        cameraPos += totalMove;
        glm::vec3 newTargetPos = _camera.getTargetPos() + totalMove;

        _camera.setCameraPos(cameraPos);
        _camera.setTargetPos(newTargetPos);
        */
    }
}

} // namespace KE

//////////////// Call backs
///////////////////////////////////////////////////////////
#include "app.hpp"
#include "backend/base/graphics_device.hpp"
#include "scene/native/prim.hpp"
#include "scene/scene_backend.hpp"
#include "ui/base_panel.hpp"
#include "utils/glm_utils.hpp"
#include "utils/print_debug.hpp"
#include "utils/asset_path.hpp"
#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <thread>

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

void App::initialize(int width, int height, bool hideUi, UpAxis upAxis,
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

    glm::vec3 cameraPos, cameraTarget;
    if (upAxis == UpAxis::Z) {
        cameraPos = glm::vec3(3.0f, 0.0f, 1.5f); // 1.5m
        cameraTarget = glm::vec3(0.0f, 0.0f, 0.85f);
    } else {
        cameraPos = glm::vec3(30.0f, 15.0f, 0.0f);
        cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    _camera.init(cameraPos, cameraTarget, upAxis);
    _camera.updateProjMatrix(_width, _height);
    _panelManager.init(this->getWindow());
    _panelManager.loadFont(KE::getAssetPath("fonts/godoFont/GodoM.ttf"), true);
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

        if (_renderVariable->deltaTime < _renderHz) {
            float remaining = _renderHz - _renderVariable->deltaTime;
            if (remaining > 0.002f) {
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<int>((remaining - 0.001f) * 1000)));
            }
            glfwPollEvents();
            continue;
        }

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

void App::setRenderHz(float renderHz) {
    _renderHz = (renderHz > 0.0f) ? renderHz : 0.0f;
}

float App::getDeltaTime() const { return _renderVariable->deltaTime; }

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
                if (buffer->prim != nullptr) {
                    // fmt::print("prim address: {}, hasTranslate: {}\n",
                    //             (void*)buffer->prim.get(),
                    //             buffer->prim->hasAttribute("xformOp:translate"));
                    auto model = buffer->prim->computeModelMatrix();
                    buffer->backendShader->setMat4("model", model);

                    auto view = this->getViewMatrix();
                    glm::mat3 normalMat =
                        glm::mat3(transpose(inverse(view * model)));
                    buffer->backendShader->setMat3("normalMat", normalMat);

                    if (std::optional<glm::vec4> color =
                            buffer->prim->getDisplayColorAlpha()) {
                        buffer->backendShader->setVec4("objColor", *color);
                    }
                }

                buffer->backendShader->setMat4("view", _viewMatrix);
                buffer->backendShader->setMat4("projection", _projectionMatrix);

                buffer->vertexArray->bind();
                _graphicsDevice->drawIndexed(buffer->numIndices);
                buffer->vertexArray->unbind();
                checkError();
            }
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

std::shared_ptr<Scene::ShapeRenderBuffer>
App::createRenderBuffer(Backend::Shader* shader,
                        const std::shared_ptr<Scene::MeshData>& meshData) {
    auto bufferInfos = std::make_shared<Scene::ShapeRenderBuffer>();
    bufferInfos->backendShader = shader;

    // VAO
    bufferInfos->vertexArray = _graphicsDevice->createVertexArray();
    bufferInfos->vertexArray->bind();

    // Index Buffer
    bufferInfos->indexBuffer = _graphicsDevice->createBuffer(
        Backend::BufferType::Index,
        sizeof(unsigned int) * meshData->indices.size(),
        meshData->indices.data());

    bufferInfos->vertexArray->setIndexBuffer(
        bufferInfos->indexBuffer.get()); // bind

    // Helper to add attribute and keep buffer alive // TODO: use one buffer
    auto addAttribute = [&](const auto& data, int location, int size) {
        if (data.empty())
            return;
        auto buffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Vertex, sizeof(data[0]) * data.size(),
            data.data());
        // buffer->bind();
        bufferInfos->vertexArray->setVertexBuffer(buffer.get());
        Backend::VertexAttribute attr = {
            location,        size, Backend::VertexAttributeType::Float, false,
            sizeof(data[0]), 0};
        bufferInfos->vertexArray->setVertexAttribute(attr);

        bufferInfos->vertexBuffers.push_back(std::move(buffer));
    };

    addAttribute(meshData->vertices, 0, 3);
    addAttribute(meshData->normals, 1, 3);
    addAttribute(meshData->uvs, 2, 2);

    bufferInfos->vertexArray->unbind();
    bufferInfos->numIndices = static_cast<int>(meshData->indices.size());

    return bufferInfos;
}

size_t App::addShape(Backend::Shader* shader,
                     std::shared_ptr<Scene::MeshData> meshData) {
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        if (shader) {
            shader->use();
        }

        auto bufferInfos = createRenderBuffer(shader, meshData);

        checkError();

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
        Backend::Shader* shader = material->getShader();

        if (shader) {
            material->bind();
        }

        auto bufferInfos = createRenderBuffer(shader, meshData);

        checkError();
        _scene->addRenderable(bufferInfos);
        return _scene->getBufferLists().size() - 1;
    } catch (const std::exception& e) {
        std::cerr << "Failed to add shape with PhongMaterial: " << e.what()
                  << std::endl;
        return static_cast<size_t>(-1);
    }
}

// Shader + Prim (non-owning: scene graph owns the Prim)
size_t App::addShape(Backend::Shader* shader, Scene::Prim* prim) {

    std::shared_ptr<Scene::MeshData> meshData = prim->getMeshData();
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        if (shader) {
            shader->use();
        }

        auto bufferInfos = createRenderBuffer(shader, meshData);
        bufferInfos->prim = prim;

        checkError();
        _scene->addRenderable(bufferInfos);
        return _scene->getBufferLists().size() - 1;

    } catch (const std::exception& e) {
        std::cerr << "Failed to add shape with Scene::Prim: " << e.what()
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
    /*
    float fov = _camera.getFoV();
    fov -= (float)yoffset;
    _camera.setFoV(fov);
    _camera.updateProjMatrix(_width, _height);
    */
    // Dolly zoom: move camera along look direction
    glm::vec3 cameraPos = _camera.getCameraPos();
    glm::vec3 cameraFront = _camera.getCameraLookDir();
    float distance = _camera.getCamToLookDistance();
    float zoomSpeed = distance * 0.05f;
    if (distance > 60 && yoffset < 0) {
        zoomSpeed = 0;
    }
    cameraPos += static_cast<float>(yoffset) * zoomSpeed * cameraFront;
    _camera.setCameraPos(cameraPos);
}

void App::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    _io->mouseX = xpos;
    _io->mouseY = ypos;
    if (_io->isMouseLeftClicked == true || _io->isMouseMiddleClicked == true ||
        _io->isMouseRightClicked == true) {
        _io->deltaMouseX += _io->mouseX - _io->prevMouseX;
        _io->deltaMouseY += _io->mouseY - _io->prevMouseY;
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
    // Capture accumulated deltas, reset for next frame.
    double deltaMouseX = _io->deltaMouseX;
    double deltaMouseY = _io->deltaMouseY;
    _io->deltaMouseX = 0;
    _io->deltaMouseY = 0;

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
    glm::vec3 globalUp = _camera.getUpVector();

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
        float dx = deltaMouseX * 0.1f;
        float dy = deltaMouseY * 0.1f;

        _camera.azimuth += (_camera.getUpAxis() == UpAxis::Z) ? -dx : dx;
        _camera.pole -= dy; // (mouse down -> cam up)

        const float EPSILON = 0.01f;
        _camera.pole = glm::clamp(_camera.pole, EPSILON, 180.0f - EPSILON);

        float phi = glm::radians(_camera.azimuth);
        float theta = glm::radians(_camera.pole);
        float r = scaleDistance;

        glm::vec3 targetPos = _camera.getTargetPos();
        glm::vec3 newCameraPos;

        if (_camera.getUpAxis() == UpAxis::Z) {
            // Z-up: pole from +Z, azimuth in XY plane
            newCameraPos.x = targetPos.x + r * glm::sin(theta) * glm::cos(phi);
            newCameraPos.y = targetPos.y + r * glm::sin(theta) * glm::sin(phi);
            newCameraPos.z = targetPos.z + r * glm::cos(theta);
        } else {
            // Y-up: pole from +Y, azimuth in XZ plane
            newCameraPos.x = targetPos.x + r * glm::sin(theta) * glm::cos(phi);
            newCameraPos.y = targetPos.y + r * glm::cos(theta);
            newCameraPos.z = targetPos.z + r * glm::sin(theta) * glm::sin(phi);
        }
        _camera.setCameraPos(newCameraPos);
    }

    if (_io->isMouseRightClicked) {
        // Look Around: Camera fixed, Target moves
        const double DRAG_MAG_THRESHOLD = 1.0;
        if (sqrt(deltaMouseX * deltaMouseX + deltaMouseY * deltaMouseY) >=
            DRAG_MAG_THRESHOLD) {

            float dx = deltaMouseX * 0.2f;
            float dy = deltaMouseY * 0.2f;

            _camera.azimuth += (_camera.getUpAxis() == UpAxis::Z) ? -dx : dx;
            _camera.pole -= dy;

            const float EPSILON = 0.01f;
            _camera.pole = glm::clamp(_camera.pole, EPSILON, 180.0f - EPSILON);

            float phi = glm::radians(_camera.azimuth);
            float theta = glm::radians(_camera.pole);
            float r = scaleDistance;

            glm::vec3 offset;
            if (_camera.getUpAxis() == UpAxis::Z) {
                offset.x = r * glm::sin(theta) * glm::cos(phi);
                offset.y = r * glm::sin(theta) * glm::sin(phi);
                offset.z = r * glm::cos(theta);
            } else {
                offset.x = r * glm::sin(theta) * glm::cos(phi);
                offset.y = r * glm::cos(theta);
                offset.z = r * glm::sin(theta) * glm::sin(phi);
            }
            _camera.setTargetPos(cameraPos - offset);
        }
    }
}

} // namespace KE

//////////////// Call backs
///////////////////////////////////////////////////////////
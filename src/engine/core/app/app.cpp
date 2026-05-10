#include "app.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/core/ui/base_panel.hpp"
#include "utils/glm_utils.hpp"
#include "utils/print_debug.hpp"
#include "utils/asset_path.hpp"
#include "utils/types.hpp"
#include <chrono>
#include <string>
#include <exception>
#include <fmt/base.h>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
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
    bool isHKeyPressed = false;
};

struct App::RenderVariable {
    float deltaTime = 0.0f;
    float lastFrameTime = 0.0f;
};

App::App()
    : _width(), _height(), _hideUI(), _window(), _camera(),
      _renderWireframe(false), _io(new App::IO),
      _renderVariable(new App::RenderVariable) {}

App::~App() {}

void App::initialize(int width, int height, bool hideUI, UpAxis upAxis,
                     Backend::BackendType graphicsBackendType,
                     Scene::BackendType sceneBackendType) {
    _width = width;
    _height = height;
    _hideUI = hideUI;
    _upAxis = upAxis;

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
    if (_window.getGlfwWindow() == nullptr) {
        std::cerr << "Failed to initialize window" << std::endl;
        return;
    }

    _logicalWidth = _window.getLogicalWidth();
    _logicalHeight = _window.getLogicalHeight();
    registerCallbacks();

    // Initialize graphics device after OpenGL context is created
    _graphicsDevice->initialize();

    _framebuffer = _graphicsDevice->createFramebuffer(
        {_width, _height, false, true,
         4}); // stencil=true for outline rendering

    glm::vec3 cameraPos, cameraTarget;
    if (_upAxis == UpAxis::Z) {
        cameraPos = glm::vec3(3.0f, 0.0f, 1.5f); // 1.5m
        cameraTarget = glm::vec3(0.0f, 0.0f, 0.85f);
    } else {
        cameraPos = glm::vec3(30.0f, 15.0f, 0.0f);
        cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    _camera.init(cameraPos, cameraTarget, _upAxis);
    _camera.updateProjMatrix(_width, _height);
    _panelManager.init(this->getWindow());
    _panelManager.loadFont(KE::getAssetPath("fonts/godoFont/GodoM.ttf"), true);
    _panelManager.addPanel(std::make_unique<MenuBarPanel>(this));
    _panelManager.addPanel(std::make_unique<ScenePanel>(this));
    _panelManager.addPanel(std::make_unique<BasePanel>());

    _graphicsDevice->setDepthTest(true);
    _graphicsDevice->setStencilTest(true);

    _rasterizer = std::make_unique<Rasterizer>(_graphicsDevice.get());
    _postProcessor = std::make_unique<PostProcessor>();
    _postProcessor->init(_graphicsDevice.get(), _width, _height);

    _rasterizer->setLight(DirectionalLight{
        (_upAxis == UpAxis::Z) ? glm::normalize(glm::vec3(0.2f, 0.5f, 1.0f))
                               : glm::normalize(glm::vec3(0.5f, 1.0f, 0.2f))});
    _initialized = true;
}

// for entire process in c++
// //////////////////////////////////////////////////////////////////////
void App::start() {
    if (!_initialized) {
        std::cerr << "App was not initialized" << std::endl;
        return;
    }

    setup();
    GLFWwindow* window = _window.getGlfwWindow();
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        float deltaTime = currentFrame - _renderVariable->lastFrameTime;
        if (deltaTime < _renderHz) {
            float remaining = _renderHz - deltaTime;
            if (remaining > 0.002f) {
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<int>((remaining - 0.001f) * 1000)));
            }
            glfwPollEvents();
            continue;
        }
        renderFrameOnce();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool App::shouldClose() {
    GLFWwindow* window = _window.getGlfwWindow();
    return window == nullptr || glfwWindowShouldClose(window);
}

void App::renderFrameOnce() {
    GLFWwindow* window = _window.getGlfwWindow();
    if (window == nullptr || glfwWindowShouldClose(window))
        return;

    float currentFrame = static_cast<float>(glfwGetTime());
    _renderVariable->deltaTime = currentFrame - _renderVariable->lastFrameTime;
    _renderVariable->lastFrameTime = currentFrame;

    processInput();
    _viewMatrix = _camera.getViewMatrix();
    _projectionMatrix = _camera.getProjMatrix();

    // ImGui::NewFrame() must come before any ImGui widget calls
    if (!_hideUI) {
        _panelManager.preRender();
    }

    this->preRender();
    if (_rasterizer) {
        _rasterizer->updateFrameData(_viewMatrix, _projectionMatrix);
        _rasterizer->renderShadowMap(_camera, _upAxis, _width, _height);
    }

    // Scene pass: render into FBO (MSAA if enabled)
    _framebuffer->bind();
    _graphicsDevice->clear(0.2f, 0.3f, 0.3f, 1.0f);
    coreRender(); // records ImGui widgets, no GL ImGui draw yet
    this->render();
    _graphicsDevice->setPolygonMode(Backend::PolygonMode::Fill);
    _framebuffer->resolve();
    _framebuffer->unbind();

    if (_postProcessor) {
        _postProcessor->process(_framebuffer->getColorTexture(), _gamma);
        _postProcessor->blitToScreen(_width, _height);
    } else {
        _framebuffer->blitToScreen(_width, _height);
    }

    // default framebuffer
    if (!_hideUI) {
        _panelManager.render();
        _panelManager.postRender();
    }
    this->postRender();
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void App::setRenderHz(float renderHz) {
    _renderHz = (renderHz > 0.0f) ? renderHz : 0.0f;
}

float App::getDeltaTime() const { return _renderVariable->deltaTime; }

void App::coreRender() {
    if (!_hideUI) {
        ImGui::Checkbox("Wireframe", &_renderWireframe);
        ImGui::SliderFloat("GammaCorrection", &_gamma, 0.f, 5.f);
        if (_rasterizer) {
            float distance = _rasterizer->getShadowDistance();
            if (ImGui::SliderFloat("Shadow Distance (Set 0 to disable shadow)",
                                   &distance, 0.0f, 100.0f)) {
                _rasterizer->setShadowDistance(distance);
            }
            int pcfSamples = _rasterizer->getShadowPcfSamples();
            if (ImGui::SliderInt("Shadow PCF Samples", &pcfSamples, 1, 16)) {
                _rasterizer->setShadowPcfSamples(pcfSamples);
            }

            auto* shadowFbo =
                _rasterizer ? _rasterizer->getShadowFbo() : nullptr;
            if (shadowFbo) {
                auto* depthTex = shadowFbo->getDepthTexture();
                if (depthTex) {
                    ImGui::Begin("Shadow Map");
                    ImGui::Text("%dx%d", depthTex->getWidth(),
                                depthTex->getHeight());
                    ImGui::Image(
                        (ImTextureID)(uintptr_t)depthTex->getNativeHandle(),
                        ImVec2(128, 128), ImVec2(0, 1),
                        ImVec2(1,
                               0)); // flip Y: GL bottom-left -> ImGui top-left
                    ImGui::End();
                }
            }
        }
    }

    if (_renderWireframe == true) {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Line);
    } else {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Fill);
    }

    if (_rasterizer)
        _rasterizer->render(_viewMatrix, _projectionMatrix);
}

void App::checkError() { _graphicsDevice->checkError(); }

MeshHandle App::addShape(Backend::Shader* shader, Scene::Prim* prim) {
    return _rasterizer ? _rasterizer->addShape(shader, prim) : InvalidHandle;
}

MeshHandle App::addShape(PhongMaterial* material, Scene::Prim* prim) {
    return _rasterizer ? _rasterizer->addShape(material, prim) : InvalidHandle;
}

void App::removePrim(MeshHandle handle, Scene::Prim* prim) {
    if (_rasterizer)
        _rasterizer->removePrim(handle, prim);
}

void App::updateShapeTransforms(MeshHandle handle,
                                const std::vector<glm::mat4>& transforms,
                                const std::vector<glm::vec4>* colors) {
    if (_rasterizer)
        _rasterizer->updateShapeTransforms(handle, transforms, colors);
}

void App::setShapeColors(MeshHandle handle,
                         const std::vector<glm::vec4>& colors) {
    if (_rasterizer)
        _rasterizer->setShapeColors(handle, colors);
}

void App::setShapeDoubleSided(MeshHandle handle, bool doubleSided) {
    if (_rasterizer)
        _rasterizer->setShapeDoubleSided(handle, doubleSided);
}

void App::setShapeTexture(MeshHandle handle, Backend::Texture* tex, int slot) {
    if (_rasterizer)
        _rasterizer->setShapeTexture(handle, tex, slot);
}

void App::updateMeshGeometry(MeshHandle handle,
                             const std::vector<glm::vec3>& positions,
                             const std::vector<glm::vec3>& normals) {
    if (_rasterizer)
        _rasterizer->updateMeshGeometry(handle, positions, normals);
}

void App::setSkybox(const std::string& path) {
    if (_rasterizer)
        _rasterizer->setSkybox(path, _upAxis);
}

void App::setSkybox(const std::vector<std::string>& paths) {
    if (_rasterizer)
        _rasterizer->setSkybox(paths, _upAxis);
}

//////////////// Call backs
///////////////////////////////////////////////////////////
void App::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    _width = width;
    _height = height;
    glfwGetWindowSize(window, &_logicalWidth, &_logicalHeight);
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

    GLFWwindow* window = this->getWindow();

    // Hide UI
    bool hPressed = glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS;
    if (hPressed && !_io->isHKeyPressed)
        _hideUI = !_hideUI;
    _io->isHKeyPressed = hPressed;

    // Prevent manipulations while ImGui using the mouse.
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }
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

glm::vec2 App::getScreenToNDC(float scrX, float scrY) {
    // screen[0, 1] space to NDC[-1, 1] in xy-plane
    float x = (scrX / _logicalWidth) * 2.0f - 1.0f;
    float y = 1.0f - (scrY / _logicalHeight) * 2.0f;
    return glm::vec2(x, y);
}

Ray App::getMouseRay() {
    glm::vec2 ndc = getScreenToNDC(_io->mouseX, _io->mouseY);
    glm::mat4 invProj = glm::inverse(_camera.getProjMatrix());
    glm::mat4 invView = glm::inverse(_camera.getViewMatrix());
    // ndc to cam space
    glm::vec4 clipNear = glm::vec4(ndc.x, ndc.y, -1.0f, 1.0f);
    glm::vec4 clipFar = glm::vec4(ndc.x, ndc.y, 1.0f, 1.0f);
    glm::vec4 camNear = invProj * clipNear;
    glm::vec4 camFar = invProj * clipFar;
    camNear /= camNear.w;
    camFar /= camFar.w;
    // cam to world
    glm::vec3 worldNear = glm::vec3(invView * camNear);
    glm::vec3 worldFar = glm::vec3(invView * camFar);
    return Ray(_camera.getCameraPos(), glm::normalize(worldFar - worldNear));
}
} // namespace KE

//////////////// Call backs
///////////////////////////////////////////////////////////

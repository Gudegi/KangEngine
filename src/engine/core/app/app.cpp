#include "app.hpp"
#include "asset/bvh_loader.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/prim_path.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/debug_draw.hpp"
#include "engine/core/ui/base_panel.hpp"
#include "utils/glm_utils.hpp"
#include "utils/print_debug.hpp"
#include "utils/asset_path.hpp"
#include "utils/types.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fmt/base.h>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace KE {
namespace {

std::string defaultMotionScenePath(const std::string& path,
                                   const Animation::SkeletonMotion& motion) {
    std::string name = motion.motionName();
    if (name.empty())
        name = std::filesystem::path(path).stem().string();
    return "/" + Scene::PrimPath::safeToken(name, "motion");
}

} // namespace

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
    bool isScreenshotKeyPressed = false; // t
    bool isEscKeyPressed = false;
    bool isSHIFTKeyPressed = false;
    bool isPickMode = false; // left click + shift
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
                     Scene::BackendType sceneBackendType, bool headless) {
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

    _window.init(_width, _height, headless);
    if (_window.getGlfwWindow() == nullptr) {
        std::cerr << "Failed to initialize window" << std::endl;
        return;
    }

    _logicalWidth = _window.getLogicalWidth();
    _logicalHeight = _window.getLogicalHeight();
    registerCallbacks();
    glfwGetFramebufferSize(_window.getGlfwWindow(), &_width, &_height);
    glfwGetWindowSize(_window.getGlfwWindow(), &_logicalWidth, &_logicalHeight);
    _uiScale.update(_window.getGlfwWindow());

    // Initialize graphics device after OpenGL context is created
    _graphicsDevice->initialize();

    Backend::FramebufferDesc sceneFramebufferDesc;
    sceneFramebufferDesc.width = _width;
    sceneFramebufferDesc.height = _height;
    sceneFramebufferDesc.stencil = true;
    sceneFramebufferDesc.msaaSamples = 4;
    sceneFramebufferDesc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
    _framebuffer = _graphicsDevice->createFramebuffer(sceneFramebufferDesc);
    _selectionMaskFramebuffer =
        _graphicsDevice->createFramebuffer({_width, _height, false, false, 0});

    glm::vec3 cameraPos, cameraTarget;
    if (_upAxis == UpAxis::Z) {
        cameraPos = glm::vec3(10.0f, 0.0f, 5.0f); // 5.0m
        cameraTarget = glm::vec3(0.0f, 0.0f, 0.85f);
    } else {
        cameraPos = glm::vec3(0.0f, 5.0f, 10.0f);
        cameraTarget = glm::vec3(0.0f, 0.85f, 0.0f);
    }
    _camera.init(cameraPos, cameraTarget, _upAxis);
    _camera.updateProjMatrix(_width, _height);
    _panelManager.init(this->getWindow());
    _panelManager.loadFont(KE::getAssetPath("fonts/godoFont/GodoM.ttf"), true);
    _panelManager.addPanel(std::make_unique<MenuBarPanel>(this));
    _panelManager.addPanel(std::make_unique<ScenePanel>(this));
    _panelManager.addPanel(std::make_unique<PerformancePanel>());
    _panelManager.addPanel(std::make_unique<RendererDebugPanel>(this));

    _graphicsDevice->setDepthTest(true);
    _graphicsDevice->setStencilTest(false);

    _rasterizer = std::make_unique<Rasterizer>(_graphicsDevice.get());
    _postProcessor = std::make_unique<PostProcessor>();
    _postProcessor->init(_graphicsDevice.get(), _width, _height);
    _selectionOutlineProcessor = std::make_unique<SelectionOutlineProcessor>();
    _selectionOutlineProcessor->init(_graphicsDevice.get(), _width, _height);
    _renderer.bind(_graphicsDevice.get(), _rasterizer.get(),
                   _postProcessor.get(), _selectionOutlineProcessor.get());
    _renderer.setViewportSize(_width, _height);

    DirectionalLight defaultLight;
    defaultLight.direction = (_upAxis == UpAxis::Z)
                                 ? glm::normalize(glm::vec3(0.2f, 0.5f, 1.0f))
                                 : glm::normalize(glm::vec3(0.5f, 1.0f, 0.2f));
    if (Scene::Prim* lightPrim = defaultDirectionalLightPrim())
        lightPrim->setDirectionalLight(defaultLight);
    else
        getRenderer().setLight(defaultLight);
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

void App::requestClose() {
    GLFWwindow* window = _window.getGlfwWindow();
    if (window)
        glfwSetWindowShouldClose(window, true);
}

void App::renderSceneToFramebuffer(Camera& camera, Backend::Framebuffer* target,
                                   int width, int height, bool clear) {
    getRenderer().syncSceneLights(getScene());
    getRenderer().renderSceneToFramebuffer(camera, target, width, height,
                                           clear);
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
        getRenderer().syncSceneLights(getScene());
        _rasterizer->updateFrameData(_viewMatrix, _projectionMatrix);
        if (_mousePickRequested) {
            const RayPickResult pick = pickMouse();
            _mousePickRequested = false;
            if (_interaction.handlePick(pick, _camera.getCameraLookDir())) {
                onForceDragBegin(_interaction.forceDragPick(),
                                 _interaction.forceDragPlanePoint());
            }
            onRayPicked(_interaction.lastPick());
        }
        if (_interaction.isForceDragActive()) {
            glm::vec3 target;
            if (_interaction.forceDragTarget(getMouseRay(), target))
                onForceDragUpdate(_interaction.forceDragPick(), target);
        }
        if (_io->isPickMode) {
            _interaction.handleHoverPick(pickMouse());
            onRayPickHover(_interaction.lastPick());
        }
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

    Backend::Texture* finalSource = _framebuffer->getColorTexture();

    if (_postProcessor && _selectionOutlineProcessor &&
        _selectionOutlineProcessor->config().enabled &&
        _selectionMaskFramebuffer && _rasterizer &&
        _interaction.hasSelection()) {
        _rasterizer->renderSelectionMaskPass(_interaction.selection(),
                                             _selectionMaskFramebuffer.get(),
                                             _width, _height);
        _selectionOutlineProcessor->renderOutlineCompositePass(
            _framebuffer->getColorTexture(),
            _selectionMaskFramebuffer->getColorTexture());
        finalSource = _selectionOutlineProcessor->getResult();
    }

    if (_postProcessor) {
        const RendererSettings& settings = getRenderer().settings();
        _postProcessor->process(finalSource, settings.gamma,
                                settings.toneMapMode, settings.toneMapExposure,
                                settings.bloom);
        _postProcessor->blitToScreen(_width, _height);
        _lastPresentedFramebuffer = _postProcessor->getOutputFramebuffer();
    } else {
        _framebuffer->blitToScreen(_width, _height);
        _lastPresentedFramebuffer = _framebuffer.get();
    }

    if (_screenshotRequested) {
        writeScreenshotFrame();
        _screenshotRequested = false;
    }

    // default framebuffer
    if (!_hideUI) {
        _panelManager.render();
        renderSelectionGizmo();
        _panelManager.postRender();
    }
    this->postRender();
    glfwSwapBuffers(window);
    glfwPollEvents();
    ++_frameIndex;
}

void App::setRenderHz(float renderHz) {
    _renderHz = (renderHz > 0.0f) ? renderHz : 0.0f;
}

std::vector<uint8_t> App::readRgbPixels(bool flipY) {
    if (_lastPresentedFramebuffer)
        return _lastPresentedFramebuffer->readColorPixels(flipY);
    if (!_framebuffer)
        return {};
    return _framebuffer->readColorPixels(flipY);
}

bool App::writePixelsPNG(const std::string& path, bool flipY) {
    auto pixels = this->readRgbPixels(flipY);
    if (pixels.empty())
        return false;
    // If fail, stbi_write_png returns 0
    return stbi_write_png(path.c_str(), _width, _height, 3, pixels.data(),
                          _width * 3) != 0;
}

bool App::writeScreenshotFrame() {
    namespace fs = std::filesystem;
    fs::path outDir = fs::path(".") / "tmp";
    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        fmt::print("Failed to create screenshot directory {}: {}\n",
                   outDir.string(), ec.message());
        return false;
    }

    const auto now = std::chrono::system_clock::now();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now.time_since_epoch()) %
                       1000;
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
    localtime_r(&nowTime, &localTime);

    std::ostringstream name;
    name << _frameIndex << "_" << std::put_time(&localTime, "%Y%m%d_%H%M%S")
         << "_" << std::setw(3) << std::setfill('0') << nowMs.count() << ".png";
    const fs::path outPath = outDir / name.str();
    const bool ok = writePixelsPNG(outPath.string(), true);
    if (ok)
        fmt::print("Saved screenshot: {}\n", outPath.string());
    else
        fmt::print("Failed to save screenshot: {}\n", outPath.string());
    return ok;
}

float App::getDeltaTime() const { return _renderVariable->deltaTime; }

void App::setCameraMoveSpeed(float speed) {
    _cameraMoveSpeed = std::max(0.0f, speed);
}

void App::coreRender() {
    if (_renderWireframe == true) {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Line);
    } else {
        _graphicsDevice->setPolygonMode(Backend::PolygonMode::Fill);
    }

    if (_rasterizer)
        _rasterizer->render(_viewMatrix, _projectionMatrix);
}

Scene::Prim* App::defaultDirectionalLightPrim() {
    if (!_scene)
        return nullptr;
    return _scene->definePrim("/lights/default_directional",
                              Scene::PrimType::Light);
}

void App::checkError() { _graphicsDevice->checkError(); }

RenderableHandle App::addRenderable(Backend::Shader* shader, Scene::Prim* prim,
                                    TransformSource transformSource) {
    return _rasterizer
               ? _rasterizer->addRenderable(shader, prim, transformSource)
               : InvalidHandle;
}

RenderableHandle
App::addSkinnedRenderable(Backend::Shader* shader, Scene::Prim* prim,
                          const Scene::SkinnedMeshData& skinnedMesh,
                          TransformSource transformSource) {
    return _rasterizer ? _rasterizer->addSkinnedRenderable(
                             shader, prim, skinnedMesh, transformSource)
                       : InvalidHandle;
}

RenderableHandle App::addRenderable(Material* material, Scene::Prim* prim,
                                    TransformSource transformSource) {
    return _rasterizer
               ? _rasterizer->addRenderable(material, prim, transformSource)
               : InvalidHandle;
}

void App::removePrim(RenderableHandle handle, Scene::Prim* prim) {
    if (_rasterizer)
        _rasterizer->removePrim(handle, prim);
}

bool App::removePrim(Scene::Prim* prim) {
    if (!prim)
        return false;
    return removePrim(prim->getPath());
}

bool App::removePrim(const std::string& path) {
    if (!_scene || path.empty() || path == "/")
        return false;

    Scene::Prim* prim = _scene->getPrimAtPath(path);
    if (!prim)
        return false;

    std::vector<Scene::Prim*> subtree;
    prim->traverse([&](Scene::Prim* child) {
        if (child)
            subtree.push_back(child);
    });

    if (_rasterizer) {
        for (Scene::Prim* child : subtree)
            _rasterizer->removePrim(child);
    }

    clearSelection();
    return _scene->removePrim(path);
}

App::MeshPrimResult App::addMeshPrim(MeshPrimDesc desc) {
    MeshPrimResult result;
    if (!desc.shader || !_scene || desc.path.empty())
        return result;

    auto* prim = _scene->definePrim(desc.path, Scene::PrimType::Mesh);
    prim->setMeshData(
        std::make_shared<Scene::MeshData>(std::move(desc.meshData)));
    prim->addTranslateOp(desc.position);
    prim->addRotateQuaternionOp(glm::normalize(desc.orientation));
    prim->addScaleOp(desc.scale);
    prim->setDisplayColorAlpha(desc.color);

    RenderableHandle handle = addRenderable(desc.shader, prim);
    if (handle != InvalidHandle) {
        setRenderableDoubleSided(handle, desc.doubleSided);
        setRenderableCastsShadow(handle, desc.castsShadow);
    }

    result.prim = prim;
    result.handle = handle;
    return result;
}

App::MeshPrimResult App::addMeshPrim(Backend::Shader* shader,
                                     const std::string& path,
                                     Scene::MeshData meshData,
                                     glm::vec3 position, glm::vec4 color,
                                     bool castsShadow) {
    MeshPrimDesc desc;
    desc.shader = shader;
    desc.path = path;
    desc.meshData = std::move(meshData);
    desc.position = position;
    desc.color = color;
    desc.castsShadow = castsShadow;
    return addMeshPrim(std::move(desc));
}

App::MeshPrimResult App::addSkinnedMeshPrim(Backend::Shader* shader,
                                            const std::string& path,
                                            Scene::SkinnedMeshData skinnedMesh,
                                            glm::vec3 position, glm::vec4 color,
                                            bool castsShadow) {
    MeshPrimResult result;
    if (!shader || !_scene || path.empty() ||
        !skinnedMesh.hasValidVertexSkinning())
        return result;

    auto* prim = _scene->definePrim(path, Scene::PrimType::Mesh);
    prim->setMeshData(std::make_shared<Scene::MeshData>(skinnedMesh.mesh));
    prim->addTranslateOp(position);
    prim->setDisplayColorAlpha(color);

    RenderableHandle handle = addSkinnedRenderable(shader, prim, skinnedMesh);
    if (handle != InvalidHandle) {
        setRenderableCastsShadow(handle, castsShadow);
    }

    result.prim = prim;
    result.handle = handle;
    return result;
}

Animation::SkeletonMotion App::loadBVHMotion(const std::string& bvhPath,
                                             float scale,
                                             const std::string& scenePath) {
    Animation::SkeletonMotion motion =
        Asset::BVHLoader::loadMotion(bvhPath, scale);
    const std::string basePath =
        scenePath.empty() ? defaultMotionScenePath(bvhPath, motion) : scenePath;
    Scene::defineSkeletonTree(getScene(), basePath + "/skeleton_tree",
                              motion.skeletonTree());
    return motion;
}

glm::vec3 App::upPos(glm::vec3 pos, UpAxis from) const {
    return axisPos(pos, from, _upAxis);
}

glm::vec3 App::upPos(float x, float y, float z, UpAxis from) const {
    return axisPos(glm::vec3(x, y, z), from, _upAxis);
}

glm::vec3 App::axisPos(float x, float y, float z, UpAxis from,
                       UpAxis to) const {
    return axisPos(glm::vec3(x, y, z), from, to);
}

glm::vec3 App::axisPos(glm::vec3 pos, UpAxis from, UpAxis to) const {
    if (from == to)
        return pos;

    // Convert 'from' to Z-Up
    glm::vec3 zUpPos = pos;
    if (from == UpAxis::X) {
        zUpPos = glm::vec3(pos.y, pos.z, pos.x);
    } else if (from == UpAxis::Y) {
        zUpPos = glm::vec3(pos.z, pos.x, pos.y);
    }

    // Convert Z-Up to 'to'
    if (to == UpAxis::X) {
        return glm::vec3(zUpPos.z, zUpPos.x, zUpPos.y);
    } else if (to == UpAxis::Y) {
        return glm::vec3(zUpPos.y, zUpPos.z, zUpPos.x);
    }
    return zUpPos;
}

glm::quat App::upQuat(glm::quat ori, UpAxis from) const {
    return axisQuat(ori, from, _upAxis);
}

glm::quat App::upQuat(float w, float x, float y, float z, UpAxis from) const {
    return axisQuat(glm::quat(w, x, y, z), from, _upAxis);
}

glm::quat App::axisQuat(float w, float x, float y, float z, UpAxis from,
                        UpAxis to) const {
    return axisQuat(glm::quat(w, x, y, z), from, to);
}

glm::quat App::axisQuat(glm::quat ori, UpAxis from, UpAxis to) const {
    if (from == to)
        return ori;

    glm::quat zUpOri = ori;
    if (from == UpAxis::X) {
        zUpOri = glm::quat(ori.w, ori.y, ori.z, ori.x);
    } else if (from == UpAxis::Y) {
        zUpOri = glm::quat(ori.w, ori.z, ori.x, ori.y);
    }

    if (to == UpAxis::X) {
        return glm::quat(zUpOri.w, zUpOri.z, zUpOri.x, zUpOri.y);
    } else if (to == UpAxis::Y) {
        return glm::quat(zUpOri.w, zUpOri.y, zUpOri.z, zUpOri.x);
    }
    return zUpOri;
}

RenderableHandle App::addCube(const std::string& path, float size,
                              glm::vec3 pos, glm::quat ori, glm::vec4 color,
                              Backend::Shader* shader) {
    MeshPrimDesc desc;
    desc.shader = shader;
    desc.path = path;
    desc.meshData = Scene::Prim::createSquareData(size);
    desc.position = pos;
    desc.orientation = ori;
    desc.color = color;
    return addMeshPrim(std::move(desc)).handle;
}

RenderableHandle App::addSphere(const std::string& path, float radius,
                                glm::vec3 pos, glm::vec4 color,
                                Backend::Shader* shader) {
    MeshPrimDesc desc;
    desc.shader = shader;
    desc.path = path;
    desc.meshData = Scene::Prim::createSphereData(radius, 32, 16);
    desc.position = pos;
    desc.color = color;
    return addMeshPrim(std::move(desc)).handle;
}

RenderableHandle App::addPlane(const std::string& path, float size,
                               glm::vec3 pos, glm::vec4 color,
                               Backend::Shader* shader) {
    MeshPrimDesc desc;
    desc.shader = shader;
    desc.path = path;
    desc.meshData = Scene::Prim::createPlaneData(size, _upAxis);
    desc.position = pos;
    desc.color = color;
    desc.doubleSided = true;
    return addMeshPrim(std::move(desc)).handle;
}

void App::drawLine(const std::string& path, glm::vec3 start, glm::vec3 end,
                   glm::vec4 color, float thickness, Backend::Shader* shader) {
    Scene::DebugDraw::logLines(this, shader, path, {start}, {end}, {color},
                               thickness);
}

void App::drawArrow(const std::string& path, glm::vec3 start, glm::vec3 end,
                    glm::vec4 color, float thickness, Backend::Shader* shader) {
    Scene::DebugDraw::logArrows(this, shader, path, {start}, {end}, {color},
                                thickness);
}

void App::setLightDirection(const glm::vec3& dir) {
    if (glm::length(dir) < 1e-4f)
        return;
    DirectionalLight light = getLight();
    light.direction = glm::normalize(dir);
    if (Scene::Prim* lightPrim = defaultDirectionalLightPrim())
        lightPrim->setDirectionalLight(light);
    else
        setLight(light);
}
void App::setLightColor(const glm::vec3& color) {
    DirectionalLight light = getLight();
    light.color = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));
    if (Scene::Prim* lightPrim = defaultDirectionalLightPrim())
        lightPrim->setDirectionalLight(light);
    else
        setLight(light);
}
void App::setLightIntensity(float intensity) {
    DirectionalLight light = getLight();
    light.intensity = std::max(0.0f, intensity);
    if (Scene::Prim* lightPrim = defaultDirectionalLightPrim())
        lightPrim->setDirectionalLight(light);
    else
        setLight(light);
}
void App::setLightAmbient(const glm::vec3& ambient) {
    DirectionalLight light = getLight();
    light.ambient = glm::clamp(ambient, glm::vec3(0.0f), glm::vec3(1.0f));
    if (Scene::Prim* lightPrim = defaultDirectionalLightPrim())
        lightPrim->setDirectionalLight(light);
    else
        setLight(light);
}

void App::updateRenderableTransforms(RenderableHandle handle,
                                     const std::vector<glm::mat4>& transforms,
                                     const std::vector<glm::vec4>* colors) {
    getRenderer().updateRenderableTransforms(handle, transforms, colors);
}

bool App::getRenderableInstanceTransform(RenderableHandle handle,
                                         int instanceIndex,
                                         glm::mat4& outTransform) const {
    return getRenderer().getRenderableInstanceTransform(handle, instanceIndex,
                                                        outTransform);
}

bool App::setRenderableInstanceTransform(RenderableHandle handle,
                                         int instanceIndex,
                                         const glm::mat4& transform) {
    return getRenderer().setRenderableInstanceTransform(handle, instanceIndex,
                                                        transform);
}

void App::updateRenderableTransforms(RenderableHandle handle,
                                     const float* transforms,
                                     const float* colors, size_t count,
                                     size_t colorCount) {
    if (count > 0 && !transforms)
        return;
    if (colorCount > 0 && !colors)
        return;
    if (colorCount != 0 && colorCount != 1 && colorCount != count)
        return;

    std::vector<glm::mat4> transformVec;
    transformVec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        glm::mat4 m(1.0f);
        std::memcpy(&m[0][0], transforms + i * 16, sizeof(float) * 16);
        transformVec.push_back(m);
    }

    std::vector<glm::vec4> colorVec;
    const std::vector<glm::vec4>* colorPtr = nullptr;
    if (colorCount > 0) {
        colorVec.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const float* c = colors + (colorCount == 1 ? 0 : i * 4);
            colorVec.emplace_back(c[0], c[1], c[2], c[3]);
        }
        colorPtr = &colorVec;
    }

    getRenderer().updateRenderableTransforms(handle, transformVec, colorPtr);
}

void App::setRenderableColors(RenderableHandle handle,
                              const std::vector<glm::vec4>& colors) {
    getRenderer().setRenderableColors(handle, colors);
}

void App::setRenderableColors(RenderableHandle handle, const float* colors,
                              size_t colorCount) {
    if (colorCount > 0 && !colors)
        return;
    std::vector<glm::vec4> colorVec;
    colorVec.reserve(colorCount);
    for (size_t i = 0; i < colorCount; ++i) {
        const float* c = colors + i * 4;
        colorVec.emplace_back(c[0], c[1], c[2], c[3]);
    }
    getRenderer().setRenderableColors(handle, colorVec);
}

void App::setRenderableDoubleSided(RenderableHandle handle, bool doubleSided) {
    getRenderer().setRenderableDoubleSided(handle, doubleSided);
}

void App::setRenderableCastsShadow(RenderableHandle handle, bool castsShadow) {
    getRenderer().setRenderableCastsShadow(handle, castsShadow);
}

void App::setRenderableTexture(RenderableHandle handle, Backend::Texture* tex,
                               TextureRole role) {
    getRenderer().setRenderableTexture(handle, tex, role);
}

void App::setRenderableTexture(RenderableHandle handle, Backend::Texture* tex,
                               int slot) {
    getRenderer().setRenderableTexture(handle, tex, slot);
}

void App::updateRenderableGeometry(RenderableHandle handle,
                                   const std::vector<glm::vec3>& positions,
                                   const std::vector<glm::vec3>& normals) {
    getRenderer().updateRenderableGeometry(handle, positions, normals);
}

void App::updateRenderableGeometry(RenderableHandle handle,
                                   const float* positions, const float* normals,
                                   size_t count, size_t normalCount) {
    if (count > 0 && !positions)
        return;
    if (normalCount > 0 && !normals)
        return;
    if (normalCount != 0 && normalCount != count)
        return;

    std::vector<glm::vec3> positionVec;
    positionVec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const float* p = positions + i * 3;
        positionVec.emplace_back(p[0], p[1], p[2]);
    }

    std::vector<glm::vec3> normalVec;
    normalVec.reserve(normalCount);
    for (size_t i = 0; i < normalCount; ++i) {
        const float* n = normals + i * 3;
        normalVec.emplace_back(n[0], n[1], n[2]);
    }

    getRenderer().updateRenderableGeometry(handle, positionVec, normalVec);
}

void App::updateRenderableSkinningMatrices(
    RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices) {
    getRenderer().updateRenderableSkinningMatrices(handle, boneMatrices);
}

void App::updateRenderableSkinningMatrices(RenderableHandle handle,
                                           const float* rowMajorMatrices,
                                           size_t count) {
    if (!rowMajorMatrices)
        return;

    std::vector<glm::mat4> matrices;
    matrices.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const float* p = rowMajorMatrices + i * 16;
        glm::mat4 m(1.0f);
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col)
                m[col][row] = p[row * 4 + col];
        }
        matrices.push_back(m);
    }
    getRenderer().updateRenderableSkinningMatrices(handle, matrices);
}

void App::logDebugLines(const std::string& path,
                        const std::vector<glm::vec3>& starts,
                        const std::vector<glm::vec3>& ends,
                        const std::vector<glm::vec4>& colors, float width,
                        bool hidden) {
    if (_rasterizer)
        _rasterizer->logDebugLines(path, starts, ends, colors, width, hidden);
}

void App::logDebugAxes(const std::string& path, const glm::mat4& transform,
                       float length, float width, bool hidden) {
    if (_rasterizer)
        _rasterizer->logDebugAxes(path, transform, length, width, hidden);
}

void App::logDebugAxes(const std::string& path, const glm::vec3& origin,
                       const glm::vec3& xAxis, const glm::vec3& yAxis,
                       const glm::vec3& zAxis, float length, float width,
                       bool hidden) {
    if (_rasterizer)
        _rasterizer->logDebugAxes(path, origin, xAxis, yAxis, zAxis, length,
                                  width, hidden);
}

void App::clearDebugLines(const std::string& path) {
    if (_rasterizer)
        _rasterizer->clearDebugLines(path);
}

void App::logDebugPoints(const std::string& path,
                         const std::vector<glm::vec3>& points,
                         const std::vector<glm::vec4>& colors, float size,
                         bool hidden) {
    if (_rasterizer)
        _rasterizer->logDebugPoints(path, points, colors, size, hidden);
}

void App::clearDebugPoints(const std::string& path) {
    if (_rasterizer)
        _rasterizer->clearDebugPoints(path);
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
    if (width <= 0 || height <= 0)
        return;

    _width = width;
    _height = height;
    glfwGetWindowSize(window, &_logicalWidth, &_logicalHeight);
    _uiScale.update(window);
    _graphicsDevice->setViewport(0, 0, _width, _height);
    _camera.updateProjMatrix(_width, _height);
    if (_framebuffer)
        _framebuffer->resize(_width, _height);
    if (_selectionMaskFramebuffer)
        _selectionMaskFramebuffer->resize(_width, _height);
    if (_postProcessor)
        _postProcessor->resize(_width, _height);
    if (_selectionOutlineProcessor)
        _selectionOutlineProcessor->resize(_width, _height);
    _renderer.setViewportSize(_width, _height);
}

void App::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    /*
    float fov = _camera.getFoV();
    fov -= (float)yoffset;
    _camera.setFoV(fov);
    _camera.updateProjMatrix(_width, _height);
    */
    // Dolly zoom: preserve the current target and scale the camera-target
    // distance, so large scenes can still zoom out without a fixed cap.
    const glm::vec3 targetPos = _camera.getTargetPos();
    const glm::vec3 cameraFront = _camera.getCameraLookDir();
    const float distance = _camera.getCamToLookDistance();
    const float zoomFactor = std::pow(0.90f, static_cast<float>(yoffset));
    const float minDistance = std::max(_camera.getNearPlane() * 2.0f, 0.01f);
    const float maxDistance =
        std::max(_camera.getFarPlane() * 0.95f, minDistance + 0.01f);
    const float nextDistance =
        glm::clamp(distance * zoomFactor, minDistance, maxDistance);
    _camera.setCameraPos(targetPos - cameraFront * nextDistance);
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
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        _io->isMouseLeftClicked = true;
        const bool shiftPressed =
            (mods & GLFW_MOD_SHIFT) != 0 ||
            glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        if (!ImGui::GetIO().WantCaptureMouse && shiftPressed) {
            _mousePickRequested = true;
            _io->isPickMode = true;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        _io->isMouseLeftClicked = false;
        _io->isPickMode = false;
        if (_interaction.isForceDragActive()) {
            _interaction.endForceDrag();
            onForceDragEnd();
        }
    }
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

    bool screenshotPressed = glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS;
    if (screenshotPressed && !_io->isScreenshotKeyPressed)
        _screenshotRequested = true;
    _io->isScreenshotKeyPressed = screenshotPressed;

    bool shiftPressed = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    _io->isSHIFTKeyPressed = shiftPressed;
    if (!shiftPressed) {
        _io->isPickMode = false;
    }

    // Prevent manipulations while ImGui using the mouse.
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    float cameraSpeed =
        static_cast<float>(_cameraMoveSpeed * _renderVariable->deltaTime);
    float scaleDistance = _camera.getCamToLookDistance();
    glm::vec3 cameraPos = _camera.getCameraPos();
    glm::vec3 cameraFront = _camera.getCameraLookDir();
    glm::vec3 cameraUp = _camera.getCameraUpDir();
    glm::vec3 cameraRight = _camera.getCameraRightDir();
    glm::vec3 globalUp = _camera.getUpVector();

    bool escPressed = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    if (escPressed && !_io->isEscKeyPressed) {
        bool endedForceDrag = false;
        if (_interaction.cancelActiveInteraction(endedForceDrag)) {
            if (endedForceDrag)
                onForceDragEnd();
        } else {
            requestClose();
        }
    }
    _io->isEscKeyPressed = escPressed;

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

    if (_io->isMouseLeftClicked && !_io->isSHIFTKeyPressed) {
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

Geometry::Ray App::getMouseRay() {
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
    return Geometry::Ray(_camera.getCameraPos(),
                         glm::normalize(worldFar - worldNear));
}

RayPickResult App::rayPick(const Geometry::Ray& ray) const {
    return _rasterizer ? _rasterizer->rayPick(ray) : RayPickResult{};
}

RayPickResult App::pickMouse() { return rayPick(getMouseRay()); }

bool App::getPickTransform(const RayPickResult& result,
                           glm::mat4& outTransform) const {
    if (!result.hit)
        return false;
    if (result.transformSource == TransformSource::SceneGraph && result.prim) {
        Scene::Prim* target = result.prim->resolveManipulationTarget();
        if (!target || !target->isActiveInHierarchy())
            return false;
        outTransform = target->computeModelMatrix();
        return true;
    }
    // ExternalBuffer transforms are owned by simulation/data streams. They are
    // pickable, but interactive manipulation should use a domain action such as
    // drag force instead of writing render instance matrices.
    return false;
}

bool App::setPickTransform(const RayPickResult& result,
                           const glm::mat4& transform) {
    if (!result.hit)
        return false;
    if (result.transformSource == TransformSource::SceneGraph && result.prim) {
        Scene::Prim* target = result.prim->resolveManipulationTarget();
        if (!target || !target->isActiveInHierarchy())
            return false;
        target->setWorldMatrix(transform);
        return true;
    }
    // ExternalBuffer transforms are simulation-owned; do not overwrite them
    // from the render gizmo.
    return false;
}

void App::renderSelectionGizmo() {
    if (_hideUI || !_interaction.hasEditableSelection())
        return;

    glm::mat4 transform(1.0f);
    if (!getPickTransform(_interaction.selection(), transform))
        return;

    if (_gizmo.manipulateTransform(_camera, transform)) {
        setPickTransform(_interaction.selection(), transform);
    }
}

} // namespace KE

//////////////// Call backs
///////////////////////////////////////////////////////////

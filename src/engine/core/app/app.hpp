///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _APP_HPP_
#define _APP_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/fwd.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstdint>
#include <vector>

#include "engine/graphics/camera/camera.hpp"
#include "engine/core/ui/panel_manager.hpp"
#include "engine/core/ui/ui_scale.hpp"
#include "utils/asset_path.hpp"
#include "geometry/ray.hpp"
#include "engine/core/app/interaction_controller.hpp"
#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/backend/graphics_factory.hpp"
#include "utils/types.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/graphics/renderer/renderer.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "engine/graphics/renderer/selection_outline_processor.hpp"
#include "engine/graphics/renderer/light.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/scene_resource_manager.hpp"
#include "engine/scene/component/scene_render_system.hpp"
#include "engine/scene/native/prim.hpp"
namespace KE {

namespace Animation {
class SkeletonMotion;
}

class ViewportPanel;

class App {
    friend class ViewportPanel;

  private:
    class GLFWCallbackWrapper { // https://stackoverflow.com/a/41089765
      private:
        static App* _app;

      public:
        GLFWCallbackWrapper() = delete;
        GLFWCallbackWrapper(const GLFWCallbackWrapper&) = delete;
        GLFWCallbackWrapper(GLFWCallbackWrapper&&) = delete;
        ~GLFWCallbackWrapper() = delete;
        static void framebufferSizeCallbackWrapper(GLFWwindow* window,
                                                   int width, int height);
        static void scrollCallbackWrapper(GLFWwindow* window, double xoffset,
                                          double yoffset);
        static void cursorPositionCallbackWrapper(GLFWwindow* window,
                                                  double xpos, double ypos);
        static void mouseButtonCallbackWrapper(GLFWwindow* window, int button,
                                               int action, int mods);
        static void setApp(App* app);
    };

  public:
    int _width, _height; // framebuffer pixels
    int _logicalWidth,
        _logicalHeight; // screen/logical pixels (matches mouse coords)
    bool _hideUI, _renderWireframe;
    bool _initialized = false;
    bool _screenshotRequested = false;
    bool _mousePickRequested = false;
    uint64_t _frameIndex = 0;
    float _cameraMoveSpeed = 15.0f;
    glm::mat4 _viewMatrix,
        _projectionMatrix; // variable to containing main camera's view and
                           // project matrix.
    UpAxis _upAxis;

    Window _window;
    Camera _camera;      // editor viewport navigation camera
    Camera _sceneCamera; // active CameraComponent mirror for
                         // viewer/overlay/hidden UI paths
    PanelManager _panelManager;
    UIScale _uiScale;

  private:
    std::unique_ptr<Backend::GraphicsDevice> _graphicsDevice;
    std::unique_ptr<Backend::Framebuffer> _framebuffer;
    std::unique_ptr<Backend::Framebuffer> _selectionMaskFramebuffer;
    std::unique_ptr<Backend::Framebuffer> _sceneCameraPreviewFramebuffer;
    std::unique_ptr<PostProcessor> _sceneCameraPreviewPostProcessor;
    int _sceneCameraPreviewPostWidth = 0;
    int _sceneCameraPreviewPostHeight = 0;
    Backend::Framebuffer* _lastPresentedFramebuffer = nullptr;
    std::unique_ptr<Scene::SceneBackend> _scene = nullptr;
    std::unique_ptr<Rasterizer> _rasterizer;
    std::unique_ptr<PostProcessor> _postProcessor;
    std::unique_ptr<SelectionOutlineProcessor> _selectionOutlineProcessor;
    Renderer _renderer;
    Scene::SceneRenderSystem _sceneRenderSystem;
    Scene::SceneResourceManager _sceneResourceManager;
    InteractionController _interaction;
    GizmoController _gizmo;
    ViewportPanel* _editorViewportPanel = nullptr;
    std::string _activeSceneCameraPath;

    Scene::Prim* defaultDirectionalLightPrim();
    Scene::Prim* activeSceneCameraPrim() const;
    bool syncActiveSceneCameraView();
    void registerCallbacks();
    bool writeScreenshotFrame();
    void renderSelectionGizmo();
    void renderSelectionGizmo(Camera& camera, const ImVec2& rectMin,
                              const ImVec2& rectSize, ImDrawList* drawList);
    void renderSelectedLightOverlay();
    void renderSelectedCameraOverlay();
    bool isEditorViewportInputActive() const;
    bool shouldBlockMouseInput() const;
    glm::vec2 getMouseNDC() const;
    bool getPickTransform(const RayPickResult& result,
                          glm::mat4& outTransform) const;
    bool setPickTransform(const RayPickResult& result,
                          const glm::mat4& transform);

  public:
    void initialize(
        int width, int height, bool hideUi, UpAxis upAxis = UpAxis::Y,
        Backend::BackendType graphicsBackendType = Backend::BackendType::OpenGL,
        Scene::BackendType sceneBackendType = Scene::BackendType::Native,
        bool headless = false);
    void processInput();
    void checkError();
    void coreRender();

    App();
    virtual ~App();

    int getWidth() { return _width; }
    int getHeight() { return _height; }
    int getLogicalWidth() { return _logicalWidth; }
    int getLogicalHeight() { return _logicalHeight; }
    const UIScale& getUiScale() const { return _uiScale; }
    UIScale& getUiScale() { return _uiScale; }

    struct IO;
    struct RenderVariable;
    std::unique_ptr<App::IO> _io;
    std::unique_ptr<App::RenderVariable> _renderVariable;
    Camera& getCamera() { return _camera; }
    Renderer& getRenderer() { return _renderer; }
    const Renderer& getRenderer() const { return _renderer; }
    Scene::SceneRenderSystem& getSceneRenderSystem() {
        return _sceneRenderSystem;
    }
    const Scene::SceneRenderSystem& getSceneRenderSystem() const {
        return _sceneRenderSystem;
    }
    Scene::SceneResourceManager& getSceneResourceManager() {
        return _sceneResourceManager;
    }
    const Scene::SceneResourceManager& getSceneResourceManager() const {
        return _sceneResourceManager;
    }
    void setLight(const DirectionalLight& light) {
        getRenderer().setLight(light);
    }
    const DirectionalLight& getLight() const { return getRenderer().light(); }
    GLFWwindow* getWindow() { return _window.getGlfwWindow(); }
    const glm::mat4& getViewMatrix() const { return _viewMatrix; }
    const glm::mat4& getProjectionMatrix() const { return _projectionMatrix; }
    Backend::GraphicsDevice* getGraphicsDevice() {
        return getRenderer().device();
    }
    Backend::Texture* getPresentedTexture();
    Rasterizer* getRasterizer() { return getRenderer().rasterizer(); }
    const Rasterizer* getRasterizer() const {
        return getRenderer().rasterizer();
    }
    SelectionOutlineProcessor* getSelectionOutlineProcessor() {
        return getRenderer().selectionOutline();
    }
    InteractionMode getInteractionMode() const { return _interaction.mode(); }
    void setInteractionMode(InteractionMode mode) {
        _interaction.setMode(mode);
    }
    void selectPrim(Scene::Prim* prim);
    bool setActiveSceneCamera(Scene::Prim* prim);
    void clearActiveSceneCamera();
    bool hasActiveSceneCamera() const {
        return !_activeSceneCameraPath.empty();
    }
    const std::string& activeSceneCameraPath() const {
        return _activeSceneCameraPath;
    }
    bool isPrimSelected(const Scene::Prim* prim) const {
        return prim && _interaction.selection().prim == prim;
    }
    bool getPrimTransformSource(const Scene::Prim* prim,
                                TransformSource& outSource) const {
        return _rasterizer &&
               _rasterizer->getPrimTransformSource(prim, outSource);
    }
    bool hasSelection() const { return _interaction.hasSelection(); }
    const RayPickResult& getSelection() const {
        return _interaction.selection();
    }
    void clearSelection() { _interaction.clearSelection(); }
    void renderSceneToFramebuffer(Camera& camera, Backend::Framebuffer* target,
                                  int width, int height, bool clear = true);
    Backend::Texture*
    renderActiveSceneCameraPreview(int width, int height,
                                   float aspectOverride = 0.0f);
    bool writeActiveSceneCameraPreviewPNG(int width, int height,
                                          float aspectOverride = 0.0f);

    //////
    void start();
    void renderFrameOnce();
    bool shouldClose();
    void requestClose();
    virtual void setup() {}      // 처음에 사용
    virtual void preRender() {}  // 루프 안에서 사용됨. 렌더 전에 사용
    virtual void render() {}     // overrideable 실제 렌더링
    virtual void postRender() {} // 렌더링 이후 마무리
    virtual void onRayPicked(const RayPickResult& result) {}
    virtual void onRayPickHover(const RayPickResult& result) {}
    virtual void onForceDragBegin(const RayPickResult& result,
                                  const glm::vec3& target) {}
    virtual void onForceDragUpdate(const RayPickResult& result,
                                   const glm::vec3& target) {}
    virtual void onForceDragEnd() {}
    //////

    // Frame rate control
    float getDeltaTime() const;
    float getMeasuredRenderFPS() const { return _measuredRenderFPS; }
    float getFrameCPUTimeMs() const { return _frameCPUTimeMs; }
    float getUpdateCPUTimeMs() const { return _updateCPUTimeMs; }
    float getRenderCPUTimeMs() const { return _renderCPUTimeMs; }
    float getPresentCPUTimeMs() const { return _presentCPUTimeMs; }
    void setVSync(bool enabled);
    bool getVSync() const;
    void setRenderHz(float renderHz);
    float getRenderHz() const { return _renderHz; }
    void setCameraMoveSpeed(float speed);
    float getCameraMoveSpeed() const { return _cameraMoveSpeed; }

  private:
    float _renderHz = 0;
    float _measuredRenderFPS = 0.0f;
    float _frameCPUTimeMs = 0.0f;
    float _updateCPUTimeMs = 0.0f;
    float _renderCPUTimeMs = 0.0f;
    float _presentCPUTimeMs = 0.0f;
    double _fpsWindowStart = 0.0;
    int _fpsWindowFrames = 0;

  public:
    virtual void framebufferSizeCallback(GLFWwindow* window, int width,
                                         int height);
    virtual void scrollCallback(GLFWwindow* window, double xoffset,
                                double yoffset);
    virtual void cursorPositionCallback(GLFWwindow* window, double xpos,
                                        double ypos);
    virtual void mouseButtonCallback(GLFWwindow* window, int button, int action,
                                     int mods);

    Scene::SceneBackend* getScene() { return _scene.get(); }
    Backend::Framebuffer* getShadowFbo() { return getRenderer().shadowFbo(); }

    RenderableHandle addRenderable(
        Backend::Shader* shader, Scene::Prim* prim,
        TransformSource transformSource = TransformSource::SceneGraph);
    RenderableHandle addSkinnedRenderable(
        Backend::Shader* shader, Scene::Prim* prim,
        const Scene::SkinnedMeshData& skinnedMesh,
        TransformSource transformSource = TransformSource::SceneGraph);
    RenderableHandle addRenderable(
        Material* material, Scene::Prim* prim,
        TransformSource transformSource = TransformSource::SceneGraph);
    RenderableHandle addSkinnedRenderable(
        Material* material, Scene::Prim* prim,
        const Scene::SkinnedMeshData& skinnedMesh,
        TransformSource transformSource = TransformSource::SceneGraph);
    void removePrim(RenderableHandle handle, Scene::Prim* prim);
    bool removePrim(Scene::Prim* prim);
    bool removePrim(const std::string& path);

    struct MeshPrimDesc {
        Backend::Shader* shader = nullptr;
        std::string path;
        Scene::MeshData meshData;
        glm::vec3 position = glm::vec3(0.0f);
        glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale = glm::vec3(1.0f);
        glm::vec4 color = glm::vec4(1.0f);
        bool doubleSided = false;
        bool castsShadow = true;
    };

    struct MeshPrimResult {
        Scene::Prim* prim = nullptr;
        RenderableHandle handle = InvalidHandle;
    };

    MeshPrimResult addMeshPrim(MeshPrimDesc desc);
    MeshPrimResult addMeshPrim(Backend::Shader* shader, const std::string& path,
                               Scene::MeshData meshData,
                               glm::vec3 position = glm::vec3(0.0f),
                               glm::vec4 color = glm::vec4(1.0f),
                               bool castsShadow = true);
    MeshPrimResult addSkinnedMeshPrim(Backend::Shader* shader,
                                      const std::string& path,
                                      Scene::SkinnedMeshData skinnedMesh,
                                      glm::vec3 position = glm::vec3(0.0f),
                                      glm::vec4 color = glm::vec4(1.0f),
                                      bool castsShadow = true);
    Animation::SkeletonMotion loadBVHMotion(const std::string& bvhPath,
                                            float scale = 1.0f,
                                            const std::string& scenePath = "");

    // Preferred renderable API. RenderableHandle identifies a renderable
    // batch/instancer; only instance transform APIs address one instance.
    void
    updateRenderableTransforms(RenderableHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors = nullptr);
    bool getRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        glm::mat4& outTransform) const;
    bool setRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        const glm::mat4& transform);
    // std::vector -> * functions for pybind
    void updateRenderableTransforms(RenderableHandle handle,
                                    const float* transforms,
                                    const float* colors, size_t count,
                                    size_t colorCount);
    void setRenderableExternalBuffer(RenderableHandle handle,
                                     const ExternalBufferDesc& desc);
    void setRenderableColors(RenderableHandle handle,
                             const std::vector<glm::vec4>& colors);
    void setRenderableColors(RenderableHandle handle, const float* colors,
                             size_t colorCount);
    void setRenderableDoubleSided(RenderableHandle handle,
                                  bool doubleSided = true);
    void setRenderableCastsShadow(RenderableHandle handle,
                                  bool castsShadow = true);
    void setRenderableAlphaMode(RenderableHandle handle, AlphaMode mode,
                                float cutoff = 0.5f);
    void setRenderableTexture(RenderableHandle handle, Backend::Texture* tex,
                              TextureRole role);
    void setRenderableTexture(RenderableHandle handle, Backend::Texture* tex,
                              int slot = 0);
    void updateRenderableGeometry(RenderableHandle handle,
                                  const std::vector<glm::vec3>& positions,
                                  const std::vector<glm::vec3>& normals);
    void updateRenderableGeometry(RenderableHandle handle,
                                  const float* positions, const float* normals,
                                  size_t count, size_t normalCount);
    void updateRenderableSkinningMatrices(
        RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices);
    void updateRenderableSkinningMatrices(RenderableHandle handle,
                                          const float* rowMajorMatrices,
                                          size_t count);

    // Debug Renderer //
    void logDebugLines(const std::string& path,
                       const std::vector<glm::vec3>& starts,
                       const std::vector<glm::vec3>& ends,
                       const std::vector<glm::vec4>& colors = {},
                       float width = 1.0f, bool hidden = false);
    void logDebugAxes(const std::string& path, const glm::mat4& transform,
                      float length = 1.0f, float width = 1.0f,
                      bool hidden = false);
    void logDebugAxes(const std::string& path, const glm::vec3& origin,
                      const glm::vec3& xAxis, const glm::vec3& yAxis,
                      const glm::vec3& zAxis, float length = 1.0f,
                      float width = 1.0f, bool hidden = false);
    void clearDebugLines(const std::string& path);
    void logDebugPoints(const std::string& path,
                        const std::vector<glm::vec3>& points,
                        const std::vector<glm::vec4>& colors = {},
                        float size = 6.0f, bool hidden = false);
    void clearDebugPoints(const std::string& path);
    ///////

    // Text Renderer //
    void setWorldText(const std::string& path, const WorldTextDesc& desc);
    void setWorldTextString(const std::string& path, std::string text);
    void setWorldTextPosition(const std::string& path,
                              const glm::vec3& position);
    void setWorldTextHidden(const std::string& path, bool hidden);
    void removeWorldText(const std::string& path);
    void clearWorldText();
    void setScreenText(const std::string& path, const ScreenTextDesc& desc);
    void setScreenTextString(const std::string& path, std::string text);
    void setScreenTextPosition(const std::string& path,
                               const glm::vec2& position);
    void setScreenTextHidden(const std::string& path, bool hidden);
    void removeScreenText(const std::string& path);
    void clearScreenText();
    ///////

    void setSkybox(const std::string& path);
    void setSkybox(const std::vector<std::string>& paths);

    glm::vec2 getScreenToNDC(float x, float y);

    // 마우스 위치로부터 3D 월드 공간의 Ray 객체 생성
    Geometry::Ray getMouseRay();
    RayPickResult rayPick(const Geometry::Ray& ray) const;
    RayPickResult pickMouse();
    const RayPickResult& getLastRayPickResult() const {
        return _interaction.lastPick();
    }

    // Coordinate Conversion Utilities
    glm::vec3 upPos(glm::vec3 pos, UpAxis from = UpAxis::Z) const;
    glm::vec3 upPos(float x, float y, float z, UpAxis from = UpAxis::Z) const;
    glm::quat upQuat(glm::quat ori, UpAxis from = UpAxis::Z) const;
    glm::quat upQuat(float w, float x, float y, float z,
                     UpAxis from = UpAxis::Z) const;
    glm::vec3 axisPos(glm::vec3 pos, UpAxis from, UpAxis to) const;
    glm::vec3 axisPos(float x, float y, float z, UpAxis from, UpAxis to) const;
    glm::quat axisQuat(glm::quat ori, UpAxis from, UpAxis to) const;
    glm::quat axisQuat(float w, float x, float y, float z, UpAxis from,
                       UpAxis to) const;

    // Debug Drawing Helpers
    void drawLine(const std::string& path, glm::vec3 start, glm::vec3 end,
                  glm::vec4 color = glm::vec4(1.0f), float thickness = 0.02f,
                  Backend::Shader* shader = nullptr);
    void drawArrow(const std::string& path, glm::vec3 start, glm::vec3 end,
                   glm::vec4 color = glm::vec4(1.0f), float thickness = 0.02f,
                   Backend::Shader* shader = nullptr);

    // Lighting & Environment Controls
    void setLightDirection(const glm::vec3& dir);
    void setLightColor(const glm::vec3& color);
    void setLightIntensity(float intensity);
    void setLightAmbient(const glm::vec3& ambient);

    // Record
    std::vector<uint8_t> readRgbPixels(bool flipY = true);
    bool writePixelsPNG(const std::string& path, bool flipY = true);
};

} // namespace KE

#endif

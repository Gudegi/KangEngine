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
#include "utils/asset_path.hpp"
#include "utils/ray.hpp"
#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/backend/graphics_factory.hpp"
#include "utils/types.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "engine/graphics/renderer/light.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/native/prim.hpp"
namespace KE {

class App {
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
    uint64_t _frameIndex = 0;
    float _gamma = 2.2; // for Gamma Correction
    float _cameraMoveSpeed = 15.0f;
    glm::mat4 _viewMatrix,
        _projectionMatrix; // variable to containing main camera's view and
                           // project matrix.
    UpAxis _upAxis;

    Window _window;
    Camera _camera;
    PanelManager _panelManager;

  private:
    std::unique_ptr<Backend::GraphicsDevice> _graphicsDevice;
    std::unique_ptr<Backend::Framebuffer> _framebuffer;
    std::unique_ptr<Scene::SceneBackend> _scene = nullptr;
    std::unique_ptr<Rasterizer> _rasterizer;
    std::unique_ptr<PostProcessor> _postProcessor;

    void registerCallbacks();
    bool writeScreenshotFrame();

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

    struct IO;
    struct RenderVariable;
    std::unique_ptr<App::IO> _io;
    std::unique_ptr<App::RenderVariable> _renderVariable;
    Camera& getCamera() { return _camera; }
    void setLight(const DirectionalLight& light) {
        if (_rasterizer)
            _rasterizer->setLight(light);
    }
    const DirectionalLight& getLight() const { return _rasterizer->getLight(); }
    GLFWwindow* getWindow() { return _window.getGlfwWindow(); }
    const glm::mat4& getViewMatrix() const { return _viewMatrix; }
    const glm::mat4& getProjectionMatrix() const { return _projectionMatrix; }
    Backend::GraphicsDevice* getGraphicsDevice() {
        return _graphicsDevice.get();
    }
    void renderSceneToFramebuffer(Camera& camera, Backend::Framebuffer* target,
                                  int width, int height,
                                  bool clear = true);

    //////
    void start();
    void renderFrameOnce();
    bool shouldClose();
    virtual void setup() {}      // 처음에 사용
    virtual void preRender() {}  // 루프 안에서 사용됨. 렌더 전에 사용
    virtual void render() {}     // overrideable 실제 렌더링
    virtual void postRender() {} // 렌더링 이후 마무리
    //////

    // Frame rate control
    float getDeltaTime() const;
    void setRenderHz(float renderHz);
    float getRenderHz() const { return _renderHz; }
    void setCameraMoveSpeed(float speed);
    float getCameraMoveSpeed() const { return _cameraMoveSpeed; }

  private:
    float _renderHz = 0;

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
    Backend::Framebuffer* getShadowFbo() {
        return _rasterizer ? _rasterizer->getShadowFbo() : nullptr;
    }

    MeshHandle addShape(Backend::Shader* shader, Scene::Prim* prim,
                        RenderTrack track = RenderTrack::SceneGraph);
    MeshHandle addSkinnedShape(Backend::Shader* shader, Scene::Prim* prim,
                               const Scene::SkinnedMeshData& skinnedMesh,
                               RenderTrack track = RenderTrack::SceneGraph);
    MeshHandle addShape(PhongMaterial* material, Scene::Prim* prim,
                        RenderTrack track = RenderTrack::SceneGraph);
    void removePrim(MeshHandle handle, Scene::Prim* prim);

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
        MeshHandle handle = InvalidHandle;
    };

    MeshPrimResult addMeshPrim(MeshPrimDesc desc);
    MeshPrimResult addMeshPrim(Backend::Shader* shader, const std::string& path,
                               Scene::MeshData meshData,
                               glm::vec3 position = glm::vec3(0.0f),
                               glm::vec4 color = glm::vec4(1.0f),
                               bool castsShadow = true);
    MeshPrimResult addSkinnedMeshPrim(
        Backend::Shader* shader, const std::string& path,
        Scene::SkinnedMeshData skinnedMesh,
        glm::vec3 position = glm::vec3(0.0f),
        glm::vec4 color = glm::vec4(1.0f), bool castsShadow = true);

    void updateShapeTransforms(MeshHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors = nullptr);
    // std::vector -> * functions for pybind
    void updateShapeTransforms(MeshHandle handle, const float* transforms,
                               const float* colors, size_t count,
                               size_t colorCount);
    void setShapeColors(MeshHandle handle,
                        const std::vector<glm::vec4>& colors);
    void setShapeColors(MeshHandle handle, const float* colors,
                        size_t colorCount);
    void setShapeDoubleSided(MeshHandle handle, bool doubleSided = true);
    void setShapeCastsShadow(MeshHandle handle, bool castsShadow = true);
    void setShapeTexture(MeshHandle handle, Backend::Texture* tex,
                         int slot = 0);
    void updateMeshGeometry(MeshHandle handle,
                            const std::vector<glm::vec3>& positions,
                            const std::vector<glm::vec3>& normals);
    void updateMeshGeometry(MeshHandle handle, const float* positions,
                            const float* normals, size_t count,
                            size_t normalCount);
    void updateSkinningMatrices(MeshHandle handle,
                                const std::vector<glm::mat4>& boneMatrices);
    void updateSkinningMatrices(MeshHandle handle, const float* rowMajorMatrices,
                                size_t count);
    void logDebugLines(const std::string& path,
                       const std::vector<glm::vec3>& starts,
                       const std::vector<glm::vec3>& ends,
                       const std::vector<glm::vec4>& colors = {},
                       float width = 1.0f, bool hidden = false);
    void clearDebugLines(const std::string& path);
    void logDebugPoints(const std::string& path,
                        const std::vector<glm::vec3>& points,
                        const std::vector<glm::vec4>& colors = {},
                        float size = 6.0f, bool hidden = false);
    void clearDebugPoints(const std::string& path);

    void setSkybox(const std::string& path);
    void setSkybox(const std::vector<std::string>& paths);

    glm::vec2 getScreenToNDC(float x, float y);

    // 마우스 위치로부터 3D 월드 공간의 Ray 객체 생성
    Ray getMouseRay();

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

    // Primitive Creation Helpers
    MeshHandle addCube(const std::string& path, float size = 1.0f,
                       glm::vec3 pos = glm::vec3(0.0f),
                       glm::quat ori = glm::quat(1, 0, 0, 0),
                       glm::vec4 color = glm::vec4(1.0f),
                       Backend::Shader* shader = nullptr);
    MeshHandle addSphere(const std::string& path, float radius = 1.0f,
                         glm::vec3 pos = glm::vec3(0.0f),
                         glm::vec4 color = glm::vec4(1.0f),
                         Backend::Shader* shader = nullptr);
    MeshHandle addPlane(const std::string& path, float size = 10.0f,
                        glm::vec3 pos = glm::vec3(0.0f),
                        glm::vec4 color = glm::vec4(1.0f),
                        Backend::Shader* shader = nullptr);

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

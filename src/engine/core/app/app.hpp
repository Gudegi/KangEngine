///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _APP_HPP_
#define _APP_HPP_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <memory>
#include <string>
#include <glm/vec3.hpp>
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
    float _gamma = 2.2; // for Gamma Correction
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

  public:
    void initialize(
        int width, int height, bool hideUi, UpAxis upAxis = UpAxis::Y,
        Backend::BackendType graphicsBackendType = Backend::BackendType::OpenGL,
        Scene::BackendType sceneBackendType = Scene::BackendType::Native);
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

    //////
    void start();
    virtual void setup() {}      // 처음에 사용
    virtual void preRender() {}  // 루프 안에서 사용됨. 렌더 전에 사용
    virtual void render() {}     // overrideable 실제 렌더링
    virtual void postRender() {} // 렌더링 이후 마무리
    //////

    // Frame rate control
    float getDeltaTime() const;
    void setRenderHz(float renderHz);
    float getRenderHz() const { return _renderHz; }

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

    MeshHandle addShape(Backend::Shader* shader, Scene::Prim* prim);
    MeshHandle addShape(PhongMaterial* material, Scene::Prim* prim);
    void removePrim(MeshHandle handle, Scene::Prim* prim);

    void updateShapeTransforms(MeshHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors = nullptr);
    void setShapeColors(MeshHandle handle,
                        const std::vector<glm::vec4>& colors);
    void setShapeDoubleSided(MeshHandle handle, bool doubleSided = true);
    void setShapeTexture(MeshHandle handle, Backend::Texture* tex,
                         int slot = 0);
    void updateMeshGeometry(MeshHandle handle,
                            const std::vector<glm::vec3>& positions,
                            const std::vector<glm::vec3>& normals);

    void setSkybox(const std::string& path);
    void setSkybox(const std::vector<std::string>& paths);

    glm::vec2 getScreenToNDC(float x, float y);

    // 마우스 위치로부터 3D 월드 공간의 Ray 객체 생성
    Ray getMouseRay();
};

} // namespace KE

#endif
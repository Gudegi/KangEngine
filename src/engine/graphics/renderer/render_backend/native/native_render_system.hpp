#pragma once

// Owns Native frame resources and execution for App rendering and camera
// previews.

#include "engine/graphics/renderer/renderer.hpp"
#include "engine/graphics/material/material.hpp"
#include "utils/types.hpp"
#include "engine/scene/render/render_scene_data.hpp"

#include <map>
#include <tuple>

namespace KE {

class NativeRenderSystem final : public RenderBackend {
  public:
    // Renderer releases offscreen targets before this system.
    NativeRenderSystem(Backend::GraphicsDevice& device, Renderer& renderer);
    ~NativeRenderSystem() override;
    NativeRenderSystem(const NativeRenderSystem&) = delete;
    NativeRenderSystem& operator=(const NativeRenderSystem&) = delete;

    RenderBackendCapabilities capabilities() const override {
        return {true, true, true, true, true, true, true};
    }
    void initialize(int width, int height) override;
    void resize(int width, int height) override;
    void syncScene(const Scene::RenderSceneFrame& scene) override;
    void syncLights(const Scene::RenderLightData& lights) override;
    RenderableHandle registerSceneBatch(
        const Scene::RenderMeshData& mesh,
        const Scene::RenderMaterialSource& material,
        const Scene::RenderInstanceData& instances,
        TransformSource source = TransformSource::SceneGraph,
        const Scene::SkinnedMeshData* skinnedMesh = nullptr) override;
    void removeSceneBatch(RenderableHandle handle) override;
    // Legacy GPU texture fields remain Native-only; nullptr removes the
    // binding.
    void bindLegacyMaterial(const Scene::RenderMaterialSource& source,
                            Material* material);
    void setSceneBatchState(RenderableHandle handle,
                            const SceneBatchState& state) override;
    void
    updateRenderableTransforms(RenderableHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors) override;
    void setRenderableExternalBuffer(RenderableHandle handle,
                                     const ExternalBufferDesc& desc) override;
    bool getRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        glm::mat4& outTransform) const override;
    bool setRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        const glm::mat4& transform) override;
    void setRenderableColors(RenderableHandle handle,
                             const std::vector<glm::vec4>& colors) override;
    void
    updateRenderableGeometry(RenderableHandle handle,
                             const std::vector<glm::vec3>& positions,
                             const std::vector<glm::vec3>& normals) override;
    void updateRenderableSkinningMatrices(
        RenderableHandle handle,
        const std::vector<glm::mat4>& boneMatrices) override;
    void syncFrame(const glm::mat4& view, const glm::mat4& projection) override;
    // Prepare shadows after scene synchronization and interaction callbacks.
    void prepareFrame(Camera& camera, UpAxis upAxis) override;
    void beginFrame() override;
    void renderScene(const glm::mat4& view, const glm::mat4& projection,
                     bool wireframe) override;
    // Call after custom scene draws; the result excludes UI overlays.
    void requestPick(const RenderPickRequest& request) override;
    std::optional<RenderPickResult> pollPick() override;
    void endFrame(const RenderPickHit& selection,
                  bool presentToScreen) override;
    Backend::Texture* renderCameraPreview(const glm::mat4& view,
                                          const glm::mat4& projection,
                                          int width, int height);

    Backend::Framebuffer* outputFramebuffer() const { return _output; }
    Backend::Framebuffer* previewFramebuffer() const;
    RenderImage readImage(bool flipY) override;
    RenderImage readImageResized(int width, int height, bool flipY) override;
    RenderImage readCameraPreview(bool flipY);
    int sampleCount() const;
    void rebuildSceneRenderTarget();
    void rebuildSceneClearTarget(const glm::vec4& clearColor);

  private:
    std::optional<RenderPickResult> _pickResult;
    void present();
    Backend::GraphicsDevice& _device;
    Renderer& _renderer;
    struct SceneBatch {
        Scene::RenderMeshData mesh;
        PBRMaterial material;
        Scene::RenderMaterialSource materialSource;
        Scene::RenderInstanceData instances;
        RenderableHandle handle = InvalidHandle;
    };
    using SceneBatchKey =
        std::tuple<Scene::ResourceHandle, Scene::ResourceHandle, bool>;
    // Rasterizer borrows these arrays/materials and is destroyed first.
    std::map<SceneBatchKey, std::unique_ptr<SceneBatch>> _sceneBatches;
    std::map<Scene::RenderInstanceId, Scene::RenderObjectSource> _sceneObjects;
    Scene::RenderLightData _sceneLighting;
    void syncSceneSnapshot(const Scene::RenderSceneFrame& scene);
    std::map<const Scene::RenderMaterialSource*, Material*> _legacyMaterials;
    std::map<RenderableHandle, const Scene::RenderMaterialSource*> _batchMaterials;
    std::map<const Scene::RenderMaterialSource*, std::unique_ptr<Material>>
        _sourceMaterials;
    void updateMaterialSource(const Scene::RenderMaterialSource& source,
                              Material& material);
    std::map<std::shared_ptr<const Scene::TextureSourceResource>,
             std::unique_ptr<Backend::Texture>>
        _materialTextures;
    std::map<Scene::ResourceHandle, std::unique_ptr<Backend::Texture>>
        _sceneTextures;
    int _width = 0;
    int _height = 0;
    // Main Scene Pipeline: RHI-owned 4x MSAA attachments resolve into the
    // legacy framebuffer's single-sample color texture during migration.
    std::unique_ptr<Backend::Framebuffer> _framebuffer;
    std::unique_ptr<Backend::Texture> _sceneMsaaColor;
    std::unique_ptr<Backend::Texture> _sceneMsaaDepthStencil;
    std::unique_ptr<Backend::TextureView> _sceneMsaaColorView;
    std::unique_ptr<Backend::TextureView> _sceneResolveColorView;
    std::unique_ptr<Backend::TextureView> _sceneMsaaDepthStencilView;
    std::unique_ptr<Backend::RenderTarget> _sceneClearTarget;
    // RHI passes append into the MSAA attachments without resolving. A final
    // empty pass performs the frame's single resolve.
    std::unique_ptr<Backend::RenderTarget> _sceneDrawTarget;
    std::unique_ptr<Backend::RenderTarget> _sceneRenderTarget;
    std::unique_ptr<Backend::Framebuffer> _selectionMaskFramebuffer;
    std::unique_ptr<Backend::Framebuffer> _previewFramebuffer;
    std::unique_ptr<PostProcessor> _previewPostProcessor;
    int _previewPostWidth = 0;
    int _previewPostHeight = 0;
    std::unique_ptr<PostProcessor> _postProcessor;
    std::unique_ptr<SelectionOutlineProcessor> _selectionOutlineProcessor;
    std::unique_ptr<Rasterizer> _rasterizer;
    Backend::Framebuffer* _output = nullptr;
};

} // namespace KE

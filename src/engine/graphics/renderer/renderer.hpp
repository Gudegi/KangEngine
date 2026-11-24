#ifndef _RENDERER_HPP_
#define _RENDERER_HPP_

#include "engine/graphics/renderer/light.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/graphics/renderer/renderer_profiler.hpp"
#include "engine/graphics/renderer/render_backend/render_image.hpp"
#include "engine/graphics/renderer/render_backend/render_backend.hpp"
#include "engine/scene/render/scene_render_bridge.hpp"
#include "engine/scene/render/render_scene_data_collector.hpp"
#include "utils/types.hpp"
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <array>
#include <memory>
#include <vector>
#include <map>
#include <tuple>
#include <unordered_map>

namespace KE {

class Camera;
class Material;
class Rasterizer;
class SelectionOutlineProcessor;
class NativeRenderSystem;
class NativeRenderExtension;

namespace Scene {
class Prim;
class SceneBackend;
struct SkinnedMeshData;
} // namespace Scene

namespace Backend {
class Framebuffer;
class GraphicsDevice;
class Texture;
class TextureView;
class RenderTarget;
class Sampler;
class BindGroupLayout;
class PipelineLayout;
class GraphicsPipeline;
class BindGroup;
} // namespace Backend

struct RendererSettings {
    float exposure = 1.0f;
    float gamma = 2.2f;
    ToneMapMode toneMapMode = ToneMapMode::None;
    float toneMapExposure = 1.0f;
    BloomConfig bloom;
    BackgroundSettings background;
};

enum class SceneHookBlendMode {
    Opaque,
    Alpha,
    Additive,
};

// Convenience descriptor for experimental draws inserted with RenderHook.
// It inherits the production scene target contract; callers only describe
// shader inputs and the render state relevant to their experiment.
struct SceneHookPipelineDesc {
    Backend::ShaderDesc shader;
    std::vector<Backend::VertexBufferLayout> vertexBuffers;
    Backend::PrimitiveTopology topology =
        Backend::PrimitiveTopology::TriangleList;
    Backend::CullMode cullMode = Backend::CullMode::None;
    SceneHookBlendMode blend = SceneHookBlendMode::Opaque;
    bool depthTest = true;
    bool depthWrite = false;
    bool useSceneFrameBindings = false;
    Backend::CompareFunction depthCompare = Backend::CompareFunction::Less;
    Backend::PipelineLayout* pipelineLayout = nullptr;
    std::string label = "scene_hook_pipeline";
};

// Owns the selected frame backend; Native resource access remains available.
class Renderer {
  public:
    Renderer();
    ~Renderer();
    void initialize(Backend::GraphicsDevice& device, int width, int height,
                    std::unique_ptr<RenderBackend> backend = nullptr);
    void initialize(std::unique_ptr<RenderBackend> backend, int width,
                    int height);
    RenderBackendCapabilities capabilities() const;
    void resizeFrame(int width, int height);
    void setBackgroundColor(const glm::vec4& color);
    void setGroundStyle(const BackgroundSettings& settings);
    void setExposure(float exposure);
    void syncScene(const Scene::RenderSceneFrame& scene);
    void syncFrame(Scene::SceneBackend* scene, const glm::mat4& view,
                   const glm::mat4& projection);
    void prepareFrame(Camera& camera, UpAxis upAxis);
    void beginFrame();
    void renderScene(const glm::mat4& view, const glm::mat4& projection,
                     bool wireframe);
    void endFrame(const RayPickResult& selection, bool presentToScreen);
    uint64_t requestPick(glm::vec2 ndc);
    std::optional<RenderPickResult> pollPick();
    void cancelPick() { _pendingPick.reset(); }
    bool hasPendingPick() const { return _pendingPick.has_value(); }
    RayPickResult resolvePick(const RenderPickHit& hit) const;
    bool buildPrimSelection(Scene::Prim& prim, RayPickResult& selection) const;
    // Explicit synchronous readback; true returns top-down rows, false
    // bottom-up.
    RenderImage readImage(bool flipY = true);
    RenderImage readImageResized(int width, int height, bool flipY = true);

    // External GPU output (currently ovrtx) for App's image presenter.
    const CudaRenderImage* cudaImage() const;

    // Native GPU output for existing viewport and camera-preview consumers.
    Backend::Texture* presentedTexture();
    Backend::Texture* renderCameraPreview(const glm::mat4& view,
                                          const glm::mat4& projection,
                                          int width, int height);
    RenderImage readCameraPreview(bool flipY = true);
    int sampleCount() const;
    void rebuildSceneRenderTarget();
    void rebuildSceneClearTarget(const glm::vec4& clearColor);
    void bind(Backend::GraphicsDevice* device, Rasterizer* rasterizer,
              PostProcessor* postProcessor,
              SelectionOutlineProcessor* selectionOutlineProcessor);
    void release();
    void setViewportSize(int width, int height);
    RendererProfiler& profiler() { return _profiler; }
    void setProfilerEnabled(bool enabled) { _profiler.setEnabled(enabled); }
    bool profilerEnabled() const { return _profiler.enabled(); }
    Backend::ProfilerCapabilities profilerCapabilities() const {
        return _device ? _device->profilerCapabilities()
                       : Backend::ProfilerCapabilities{};
    }
    RendererProfiler::Snapshot latestFrameProfile() const {
        return _profiler.latestFrameProfile();
    }
    std::vector<RendererProfiler::Snapshot> frameProfileHistory() const {
        return _profiler.frameProfileHistory();
    }
    Backend::ProfileSummary
    profileSummary(size_t maxFrames = RendererProfiler::HistoryCapacity) const {
        return _profiler.profileSummary(maxFrames);
    }
    void exportProfileJson(
        const std::string& path,
        size_t maxFrames = RendererProfiler::HistoryCapacity) const {
        _profiler.exportJson(path, maxFrames);
    }

    // Native-only access; throws for external backends. Legacy accessors
    // remain.
    NativeRenderExtension& native();
    const NativeRenderExtension& native() const;
    Backend::GraphicsDevice* device() { return _device; }
    const Backend::GraphicsDevice* device() const { return _device; }
    Rasterizer* rasterizer() { return _rasterizer; }
    const Rasterizer* rasterizer() const { return _rasterizer; }
    PostProcessor* postProcessor() { return _postProcessor; }
    const PostProcessor* postProcessor() const { return _postProcessor; }
    SelectionOutlineProcessor* selectionOutline() {
        return _selectionOutlineProcessor;
    }
    const SelectionOutlineProcessor* selectionOutline() const {
        return _selectionOutlineProcessor;
    }
    RendererSettings& settings() { return _settings; }
    // Catalog outlives this renderer's registered scene batches.
    void bindSceneResources(const Scene::SceneResourceManager& resources);
    const RendererSettings& settings() const { return _settings; }
    void applyBackgroundSettings();

    // Evaluated world-space inputs. Directions follow emitted rays.
    void setLight(const DirectionalLight& light);
    const DirectionalLight& light() const;
    void setPointLights(std::vector<PointLight> lights);
    const std::vector<PointLight>& pointLights() const;
    void setSpotLights(std::vector<SpotLight> lights);
    const std::vector<SpotLight>& spotLights() const;
    void syncSceneLights(Scene::SceneBackend* scene);
    Backend::Framebuffer* shadowFbo();
    RenderHookHandle addRenderHook(RenderHookPhase phase,
                                   RenderHookCallback callback);
    bool removeRenderHook(RenderHookHandle handle);
    std::unique_ptr<Backend::GraphicsPipeline>
    createSceneHookPipeline(const SceneHookPipelineDesc& desc);
    void renderSceneToFramebuffer(Camera& camera, Backend::Framebuffer* target,
                                  int width, int height, bool clear = true);
    void renderSceneToFramebuffer(const glm::mat4& view, const glm::mat4& proj,
                                  Backend::Framebuffer* target, int width,
                                  int height, bool clear = true);

    RenderableHandle addRenderable(
        Material* material, Scene::Prim* prim,
        TransformSource transformSource = TransformSource::SceneGraph);
    RenderableHandle addSkinnedRenderable(
        Material* material, Scene::Prim* prim,
        const Scene::SkinnedMeshData& skinnedMesh,
        TransformSource transformSource = TransformSource::SceneGraph);
    void removePrim(RenderableHandle handle, Scene::Prim* prim);
    void removePrim(Scene::Prim* prim);

    // RenderableHandle identifies a renderable batch/instancer. Most controls
    // apply to the whole batch; APIs with instanceIndex can target one
    // instance.
    void
    updateRenderableTransforms(RenderableHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors = nullptr);
    void setRenderableExternalBuffer(RenderableHandle handle,
                                     const ExternalBufferDesc& desc);
    std::vector<Sim::GpuArrayView> mapRenderableCudaTransformBuffers(
        const std::vector<RenderableHandle>& handles, int count, int deviceId,
        uint64_t streamHandle);
    void unmapRenderableCudaTransformBuffers(
        const std::vector<RenderableHandle>& handles, int deviceId,
        uint64_t streamHandle);
    bool getRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        glm::mat4& outTransform) const;
    bool setRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        const glm::mat4& transform);
    void setRenderableColors(RenderableHandle handle,
                             const std::vector<glm::vec4>& colors);
    void setRenderableDoubleSided(RenderableHandle handle,
                                  bool doubleSided = true);
    void setSceneBatchState(RenderableHandle handle,
                            const SceneBatchState& state);
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
    void updateRenderableSkinningMatrices(
        RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices);

  private:
    using SceneBatchKey =
        std::tuple<const Scene::MeshData*, Material*, TransformSource, size_t>;
    struct SceneBatch {
        SceneBatchKey key;
        Scene::RenderMeshData mesh;
        Material* material = nullptr;
        Scene::RenderMaterialSource materialSource;
        Scene::SceneRenderBridge bridge;
        SceneBatchState state;
        bool hasGeometryUpdates = false;
        RenderableHandle handle = InvalidHandle;
    };

    RenderPickRequest _pickView;
    std::optional<RenderPickRequest> _pendingPick;
    uint64_t _nextPickId = 1;
    uint64_t _instanceRevision = 0;
    uint64_t _pickInstanceRevision = 0;

    size_t _nextSkinningId = 1;
    // Backend draws borrow these arrays; release the backend before the
    // sources.
    std::multimap<SceneBatchKey, std::unique_ptr<SceneBatch>> _sceneBatches;
    std::unordered_map<RenderableHandle, SceneBatch*> _sceneBatchHandles;

    // Native extension resources are released before the frame backend.
    std::unique_ptr<RenderBackend> _renderBackend;
    std::unique_ptr<NativeRenderExtension> _native;
    RenderBackendCapabilities _backendCapabilities;
    Scene::RenderLightCollector _lightCollector;
    bool _snapshotLighting = false;
    bool _sceneLighting = false;
    const Scene::SceneResourceManager* _sceneResources = nullptr;
    RendererProfiler _profiler;
    // Borrowed aliases for existing Native accessors; owned by the backend.
    Backend::GraphicsDevice* _device = nullptr;
    Rasterizer* _rasterizer = nullptr;
    PostProcessor* _postProcessor = nullptr;
    SelectionOutlineProcessor* _selectionOutlineProcessor = nullptr;
    RendererSettings _settings;

    RenderBackend& renderBackend();
    NativeRenderSystem& nativeRenderSystem();
    RenderableHandle
    addSceneRenderable(Material& material, Scene::Prim& prim,
                       TransformSource source,
                       const Scene::SkinnedMeshData* skinnedMesh);
    void releaseEmptyBatch(RenderableHandle handle);
    void syncSceneInstances();
};

} // namespace KE

#endif

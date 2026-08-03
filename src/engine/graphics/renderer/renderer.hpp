#ifndef _RENDERER_HPP_
#define _RENDERER_HPP_

#include "engine/graphics/renderer/light.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <array>
#include <memory>
#include <vector>

namespace KE {

class Camera;
class Material;
class Rasterizer;
class SelectionOutlineProcessor;

namespace Scene {
class Prim;
class SceneBackend;
struct SkinnedMeshData;
} // namespace Scene

namespace Backend {
class Framebuffer;
class GraphicsDevice;
class Shader;
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

// Facade for render-system access. App owns the concrete resources; Renderer
// only groups the public rendering surface so App does not expose every member
// directly as the primary API.
class Renderer {
  public:
    ~Renderer();
    void bind(Backend::GraphicsDevice* device, Rasterizer* rasterizer,
              PostProcessor* postProcessor,
              SelectionOutlineProcessor* selectionOutlineProcessor);
    void setViewportSize(int width, int height);

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
    const RendererSettings& settings() const { return _settings; }
    void setBackgroundShader(Backend::Shader* shader);
    Backend::Shader* backgroundShader() const { return _backgroundShader; }
    void applyBackgroundSettings();

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
    void ensureOffscreenSceneTarget(Backend::Framebuffer* target, int width,
                                    int height);
    void ensureOffscreenFormatConversion(Backend::Texture* targetColor);
    void recordOffscreenFormatConversion(int width, int height);
    Backend::GraphicsDevice* _device = nullptr;
    Rasterizer* _rasterizer = nullptr;
    PostProcessor* _postProcessor = nullptr;
    SelectionOutlineProcessor* _selectionOutlineProcessor = nullptr;
    Backend::Framebuffer* _offscreenFramebuffer = nullptr;
    Backend::Texture* _offscreenResolvedColor = nullptr;
    int _offscreenWidth = 0;
    int _offscreenHeight = 0;
    std::unique_ptr<Backend::Texture> _offscreenMsaaColor;
    std::unique_ptr<Backend::Texture> _offscreenIntermediateColor;
    std::unique_ptr<Backend::Texture> _offscreenMsaaDepthStencil;
    std::unique_ptr<Backend::TextureView> _offscreenMsaaColorView;
    std::unique_ptr<Backend::TextureView> _offscreenResolveColorView;
    std::unique_ptr<Backend::TextureView> _offscreenDepthStencilView;
    std::unique_ptr<Backend::RenderTarget> _offscreenDrawTarget;
    std::unique_ptr<Backend::RenderTarget> _offscreenResolveTarget;
    std::unique_ptr<Backend::TextureView> _offscreenOutputView;
    std::unique_ptr<Backend::RenderTarget> _offscreenOutputTarget;
    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4>
        _offscreenCopyGroupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _offscreenCopyPipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _offscreenCopyPipeline;
    std::unique_ptr<Backend::Sampler> _offscreenCopySampler;
    std::unique_ptr<Backend::BindGroup> _offscreenCopyBindGroup;
    Backend::TextureFormat _offscreenOutputFormat =
        Backend::TextureFormat::Undefined;
    std::unique_ptr<Backend::PipelineLayout> _sceneHookDefaultPipelineLayout;
    std::unique_ptr<Backend::PipelineLayout> _sceneHookFramePipelineLayout;
    Backend::Shader* _backgroundShader = nullptr;
    RendererSettings _settings;
    int _viewportWidth = 0;
    int _viewportHeight = 0;
};

} // namespace KE

#endif

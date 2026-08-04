#ifndef _RASTERIZER_HPP_
#define _RASTERIZER_HPP_

#include "engine/graphics/renderer/render_pipeline.hpp"
#include "engine/graphics/renderer/selection_mask_pass.hpp"
#include "engine/graphics/renderer/forward_pass.hpp"
#include "engine/graphics/renderer/shadow_pass.hpp"
#include "engine/graphics/renderer/skybox_pass.hpp"
#include "engine/graphics/renderer/debug_renderer.hpp"
#include "engine/graphics/renderer/text_renderer.hpp"
#include "engine/graphics/renderer/mesh_instancer.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/graphics/renderer/light.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/graphics/camera/camera.hpp"
#include "geometry/bounds.hpp"
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include "utils/types.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace KE {

// ---------------------------------------------------------------------------
// Two-source transform ownership
//
// Source A — SceneGraph-driven
//   addRenderable(material, prim) → instancer polls Prim attributes every frame
//   Use for: static geometry, any object the scene graph owns
//
// Source B — ExternalBuffer-driven
//   handle = addRenderable(material, prim)     // register once at setup
//   setRenderableColors(handle, colors)        // upload colors once
//   updateRenderableTransforms(handle, mats)   // upload transforms per frame
//   Use for: PhysX rigid bodies, large instanced crowds, anything with
//            per-frame external transform arrays
//
// Sources are split at the Rasterizer key level, so scene-graph objects and
// external simulation buffers can share shader/mesh/material without
// interfering. Legacy shader-only APIs are wrapped into VertexColorMaterial
// before they reach this layer.
// ---------------------------------------------------------------------------

class Rasterizer : public RenderPipeline {
  public:
    static constexpr int MaxShadowCascades = 4;

  private:
    // =====================================================================
    // Scene Mesh Pipelines - batching and renderable registration
    // =====================================================================
    struct InstancerKey {
        const Scene::MeshData* mesh;
        Material* material;
        TransformSource transformSource;
        bool operator<(const InstancerKey& o) const {
            if (mesh != o.mesh)
                return mesh < o.mesh;
            if (material != o.material)
                return material < o.material;
            return transformSource < o.transformSource;
        }
    };
    std::map<InstancerKey, MeshInstancer> _instancers;

    std::map<InstancerKey, RenderableHandle> _handleMap;
    std::vector<MeshInstancer*> _handleTable;
    struct PrimSourceRegistrations {
        // To specify which track is in the Scene panel
        uint32_t sceneGraph = 0;
        uint32_t external = 0;
    };
    std::unordered_map<const Scene::Prim*, PrimSourceRegistrations>
        _primSourceRegistrations;
    void registerPrimSource(Scene::Prim* prim, TransformSource source);
    void unregisterPrimSource(Scene::Prim* prim, TransformSource source);

    struct RenderHookEntry {
        RenderHookHandle handle = InvalidRenderHook;
        RenderHookCallback callback;
    };
    std::array<std::vector<RenderHookEntry>, 2> _renderHooks;
    RenderHookHandle _nextRenderHook = 1;

    // =====================================================================
    // Debug Overlay / Text Pipelines
    // =====================================================================
    DebugRenderer _debugRenderer;
    TextRenderer _textRenderer;
    int _viewportWidth = 1;
    int _viewportHeight = 1;

    // =====================================================================
    // Shared Frame Data - camera and lighting
    // =====================================================================
    std::unique_ptr<Backend::Buffer> _cameraUBO;
    std::unique_ptr<Backend::Buffer> _lightUBO;
    std::unique_ptr<Backend::Buffer> _shadowUBO;
    DirectionalLight _light;
    std::vector<PointLight> _pointLights;
    std::vector<SpotLight> _spotLights;
    bool _lightDirty = true;

    // =====================================================================
    // Shadow Depth Pipeline - depth targets, CSM, and immutable variants
    // =====================================================================
    std::unique_ptr<Backend::Framebuffer> _shadowFbo; // depth-only
    int _shadowMapWH = 4096;
    std::array<int, MaxShadowCascades> _cascadeMapSizes{4096, 2048, 1024, 1024};
    std::array<std::unique_ptr<Backend::Framebuffer>, MaxShadowCascades>
        _cascadeFbos;
    std::array<Backend::Texture*, MaxShadowCascades> _cascadeMaps{};
    ShadowPass _shadowPass;

    // =====================================================================
    // Opaque Vertex-Color Forward Pipeline - first main-scene RHI path
    // =====================================================================
    ForwardPass _forwardPass;

    // =====================================================================
    // Skybox Pipeline - cubemap resource and immutable background draw
    // =====================================================================
    SkyboxPass _skyboxPass;

    // =====================================================================
    // Selection Mask Pipeline - opaque, alpha-mask, and skinned variants
    // =====================================================================
    SelectionMaskPass _selectionMaskPass;

    // Shadow configuration and per-frame cascade state.
    float _shadowRadius = 3.0f;
    int _shadowPcfSamples = 16;
    float _shadowDistance = 20.0f; // 0 = shadow disabled
    float _activeShadowOrthoHalfSize = 0.0f;
    glm::mat4 _lightSpaceMatrix{1.f};
    Backend::Texture* _shadowMap = nullptr;
    // CSM
    bool _useCsm = false;
    bool _debugCsmCascadeTint = false;
    bool _useTightShadowFit = true;
    int _cascadeCount = 3;
    // Blends uniform and logarithmic cascade splits: 0=uniform, 1=log.
    float _cascadeLambda = 0.55f;
    std::array<float, MaxShadowCascades> _cascadeSplits{};
    std::array<float, MaxShadowCascades> _cascadeOrthoHalfSizes{};
    std::array<glm::mat4, MaxShadowCascades> _cascadeLightMatrices{};

    // =====================================================================
    // Frustum Culling / Renderer Diagnostics
    // =====================================================================
    Geometry::Frustum _viewFrustum;
    bool _frustumCullingEnabled = true;
    bool _wireframeEnabled = false;
    bool _debugRenderAABB = false;
    int _cullingTotalBatches = 0;
    int _cullingCulledBatches = 0;
    int _cullingTotalInstances = 0;
    int _cullingCulledInstances = 0;

    // =====================================================================
    // Shadow Depth Pipeline - private operations
    // =====================================================================
    void updateShadowUBO(float activeOrthoHalfSize);
    void initShadowRhi();
    void rebuildShadowSamplingBindings(Backend::Texture* fallbackTexture);
    void initForwardRhi();

    // =====================================================================
    // Selection Mask Pipeline - private operations
    // =====================================================================

    // Shadow matrix/caster operations.
    void updateShadowPassUBO(const glm::mat4& lightSpaceMatrix,
                             float activeOrthoHalfSize);
    void setShadowMap(Backend::Texture* tex, const glm::mat4& lightSpaceMat,
                      float radius, float distance);
    void drawShadowCasters(Backend::RenderTarget* target, int mapSize);

    // =====================================================================
    // Scene / Skybox / Transparent / Debug Pipelines - private operations
    // =====================================================================
    void updateDebugRenderAABB();
    Backend::Texture* activeShadowTexture() const;
    void bindShadowTextures(Backend::Texture* shadowTexture);
    void renderOpaquePass(Backend::Texture* shadowTexture,
                          Backend::RenderTarget* sceneDrawTarget);
    void renderSkyboxPass(const glm::mat4& view, const glm::mat4& proj,
                          Backend::RenderTarget* sceneDrawTarget);
    void renderTransparentPass(Backend::Texture* shadowTexture,
                               Backend::RenderTarget* sceneDrawTarget);
    void renderDebugOverlayPass(Backend::RenderTarget* sceneDrawTarget);
    void recordRenderHooks(RenderHookPhase phase,
                           Backend::RenderTarget* sceneDrawTarget);

  public:
    Rasterizer(Backend::GraphicsDevice* graphicsDevice);
    Backend::BindGroupLayout* sceneFrameBindGroupLayout() const {
        return _forwardPass.frameLayout();
    }
    RenderHookHandle addRenderHook(RenderHookPhase phase,
                                   RenderHookCallback callback);
    bool removeRenderHook(RenderHookHandle handle);

    // =====================================================================
    // Lighting API
    // =====================================================================
    void setLight(const DirectionalLight& light) {
        _light = light;
        _lightDirty = true;
    }
    const DirectionalLight& getLight() const { return _light; }
    void setPointLights(std::vector<PointLight> lights) {
        if (lights.size() > MaxPointLights)
            lights.resize(MaxPointLights);
        _pointLights = std::move(lights);
        _lightDirty = true;
    }
    const std::vector<PointLight>& getPointLights() const {
        return _pointLights;
    }
    void setSpotLights(std::vector<SpotLight> lights) {
        if (lights.size() > MaxSpotLights)
            lights.resize(MaxSpotLights);
        _spotLights = std::move(lights);
        _lightDirty = true;
    }
    const std::vector<SpotLight>& getSpotLights() const { return _spotLights; }

    // =====================================================================
    // Shadow Depth / CSM API
    // =====================================================================
    void setShadowDistance(float distance) {
        _shadowDistance = std::max(0.0f, distance);
    }
    float getShadowDistance() const { return _shadowDistance; }
    void setShadowPcfSamples(int samples) {
        _shadowPcfSamples = std::clamp(samples, 1, 16);
        updateShadowUBO(_shadowMap ? _activeShadowOrthoHalfSize : 0.0f);
    }
    int getShadowPcfSamples() const { return _shadowPcfSamples; }
    void setUseCsm(bool enabled) { _useCsm = enabled; }
    bool getUseCsm() const { return _useCsm; }
    void setDebugCsmCascadeTint(bool enabled) {
        _debugCsmCascadeTint = enabled;
    }
    bool getDebugCsmCascadeTint() const { return _debugCsmCascadeTint; }
    void setUseTightShadowFit(bool enabled) { _useTightShadowFit = enabled; }
    bool getUseTightShadowFit() const { return _useTightShadowFit; }
    void setCascadeCount(int count) {
        _cascadeCount = std::clamp(count, 1, MaxShadowCascades);
    }
    int getCascadeCount() const { return _cascadeCount; }
    void setCascadeLambda(float lambda) {
        _cascadeLambda = std::clamp(lambda, 0.0f, 1.0f);
    }
    float getCascadeLambda() const { return _cascadeLambda; }
    const std::array<float, MaxShadowCascades>& getCascadeSplits() const {
        return _cascadeSplits;
    }
    Backend::Framebuffer* getCascadeShadowFbo(int index) {
        if (index < 0 || index >= _cascadeCount)
            return nullptr;
        return _cascadeFbos[static_cast<size_t>(index)].get();
    }
    void renderShadowMap(Camera& camera, UpAxis upAxis, int viewportWidth,
                         int viewportHeight);
    glm::mat4 computeLightSpaceMatrix(Camera& camera, const UpAxis upAxis);
    std::array<float, MaxShadowCascades> computeCascadeSplits(Camera& camera);
    glm::mat4 computeLightSpaceMatrix(Camera& camera, const UpAxis upAxis,
                                      float shadowNear, float shadowFar);
    Backend::Framebuffer* getShadowFbo() { return _shadowFbo.get(); }

    // =====================================================================
    // Culling / Renderer Diagnostics API
    // =====================================================================
    void setFrustumCullingEnabled(bool enabled) {
        _frustumCullingEnabled = enabled;
    }
    void setWireframeEnabled(bool enabled) { _wireframeEnabled = enabled; }
    bool isFrustumCullingEnabled() const { return _frustumCullingEnabled; }
    void setBackgroundSettings(const BackgroundSettings& settings);
    void setDebugRenderAABB(bool enabled) { _debugRenderAABB = enabled; }
    bool getDebugRenderAABB() const { return _debugRenderAABB; }
    int getCullingTotalBatches() const { return _cullingTotalBatches; }
    int getCullingCulledBatches() const { return _cullingCulledBatches; }
    int getCullingTotalInstances() const { return _cullingTotalInstances; }
    int getCullingCulledInstances() const { return _cullingCulledInstances; }

    // =====================================================================
    // Renderable Registration / Simulation Buffer API
    // =====================================================================
    RenderableHandle addRenderable(
        Material* material, Scene::Prim* prim,
        TransformSource transformSource = TransformSource::SceneGraph);
    RenderableHandle addSkinnedRenderable(
        Material* material, Scene::Prim* prim,
        const Scene::SkinnedMeshData& skinnedMesh,
        TransformSource transformSource = TransformSource::SceneGraph);
    void removePrim(RenderableHandle handle, Scene::Prim* prim);
    void removePrim(Scene::Prim* prim);

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

    void setRenderableColors(RenderableHandle handle,
                             const std::vector<glm::vec4>& colors);

    // Disable back-face culling for this instancer (e.g. cloth, thin surfaces).
    void setRenderableDoubleSided(RenderableHandle handle,
                                  bool doubleSided = true);
    void setRenderableCastsShadow(RenderableHandle handle,
                                  bool castsShadow = true);
    // Mask performs alpha cutoff in opaque/depth passes; Blend selects the
    // transparent pass. Cutoff is ignored by Opaque and Blend.
    void setRenderableAlphaMode(RenderableHandle handle, AlphaMode mode,
                                float cutoff = 0.5f);
    void setRenderableTexture(RenderableHandle handle, Backend::Texture* tex,
                              TextureRole role);
    void setRenderableTexture(RenderableHandle handle, Backend::Texture* tex,
                              int slot = 0);
    RayPickResult rayPick(const Geometry::Ray& ray) const;
    bool getRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        glm::mat4& outTransform) const;
    bool setRenderableInstanceTransform(RenderableHandle handle,
                                        int instanceIndex,
                                        const glm::mat4& transform);

    // Deformable mesh: update vertex positions + normals each frame.
    void updateRenderableGeometry(RenderableHandle handle,
                                  const std::vector<glm::vec3>& positions,
                                  const std::vector<glm::vec3>& normals);
    void updateRenderableSkinningMatrices(
        RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices);

    // =====================================================================
    // Debug Overlay API
    // =====================================================================
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

    // =====================================================================
    // Viewport / Skybox API
    // =====================================================================
    void setViewportSize(int width, int height) {
        _viewportWidth = std::max(width, 1);
        _viewportHeight = std::max(height, 1);
    }

    void setSkybox(const std::string& path, UpAxis upAxis = UpAxis::Y);
    void setSkybox(const std::vector<std::string>& paths,
                   UpAxis upAxis = UpAxis::Y);

    // =====================================================================
    // Frame Rendering / Selection Pipeline API
    // =====================================================================
    void updateFrameData(const glm::mat4& view, const glm::mat4& proj);
    void render(const glm::mat4& view, const glm::mat4& proj,
                Backend::RenderTarget* sceneDrawTarget) override;
    bool buildPrimSelection(Scene::Prim* prim,
                            RayPickResult& outSelection) const;
    bool getPrimTransformSource(const Scene::Prim* prim,
                                TransformSource& outSource) const;
    void renderSelectionMaskPass(const RayPickResult& selection,
                                 Backend::Framebuffer* target, int width,
                                 int height);
};

} // namespace KE

#endif

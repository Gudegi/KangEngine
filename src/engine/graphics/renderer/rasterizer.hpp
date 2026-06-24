#ifndef _RASTERIZER_HPP_
#define _RASTERIZER_HPP_

#include "engine/graphics/renderer/render_pipeline.hpp"
#include "engine/graphics/renderer/debug_renderer.hpp"
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
#include <utility>
#include <vector>

namespace KE {

// ---------------------------------------------------------------------------
// Two-source transform ownership
//
// Source A — SceneGraph-driven
//   addRenderable(shader, prim) → instancer polls Prim attributes every frame
//   Use for: static geometry, any object the scene graph owns
//
// Source B — ExternalBuffer-driven
//   handle = addRenderable(shader, prim)       // register once at setup
//   setRenderableColors(handle, colors)        // upload colors once
//   updateRenderableTransforms(handle, mats)   // upload transforms per frame
//   Use for: PhysX rigid bodies, large instanced crowds, anything with
//            per-frame external transform arrays
//
// Sources are split at the Rasterizer key level, so scene-graph objects and
// external simulation buffers can share shader/mesh/material without
// interfering.
// ---------------------------------------------------------------------------

class Rasterizer : public RenderPipeline {
  public:
    static constexpr int MaxShadowCascades = 4;

  private:
    // Prim-based instanced rendering
    struct InstancerKey {
        Backend::Shader* shader;
        const Scene::MeshData* mesh;
        Material* material; // nullptr for shader-only path
        TransformSource transformSource;
        bool operator<(const InstancerKey& o) const {
            if (shader != o.shader)
                return shader < o.shader;
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
    DebugRenderer _debugRenderer;

    std::unique_ptr<Backend::Buffer> _cameraUBO;
    std::unique_ptr<Backend::Buffer> _lightUBO;
    std::unique_ptr<Backend::Buffer> _shadowUBO;
    DirectionalLight _light;
    std::vector<PointLight> _pointLights;
    std::vector<SpotLight> _spotLights;
    bool _lightDirty = true;

    std::unique_ptr<Backend::Framebuffer> _shadowFbo; // depth-only
    int _shadowMapWH = 4096;
    std::array<int, MaxShadowCascades> _cascadeMapSizes{4096, 2048, 1024, 1024};
    std::array<std::unique_ptr<Backend::Framebuffer>, MaxShadowCascades>
        _cascadeFbos;
    std::array<Backend::Texture*, MaxShadowCascades> _cascadeMaps{};
    std::unique_ptr<Backend::Shader> _shadowShader;
    std::unique_ptr<Backend::Shader> _skinnedShadowShader;
    std::unique_ptr<Backend::Shader> _selectionMaskShader;
    std::unique_ptr<Backend::Shader> _skinnedSelectionMaskShader;
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
    Geometry::Frustum _viewFrustum;
    bool _frustumCullingEnabled = true;
    bool _debugRenderAABB = false;
    int _cullingTotalBatches = 0;
    int _cullingCulledBatches = 0;
    int _cullingTotalInstances = 0;
    int _cullingCulledInstances = 0;

    // shadow
    void updateShadowUBO(float activeOrthoHalfSize);
    void updateShadowPassUBO(const glm::mat4& lightSpaceMatrix,
                             float activeOrthoHalfSize);
    void setShadowMap(Backend::Texture* tex, const glm::mat4& lightSpaceMat,
                      float radius, float distance);
    void drawShadowCasters();
    void updateDebugRenderAABB();
    Backend::Texture* activeShadowTexture() const;
    void bindShadowTextures(Backend::Texture* shadowTexture);
    void bindShadowSampler(Backend::Shader* shader,
                           Backend::Texture* shadowTexture);
    void renderSceneInstancer(MeshInstancer& inst, bool transparentPass,
                              Backend::Texture* shadowTexture);
    void renderOpaquePass(Backend::Texture* shadowTexture);
    void renderSkyboxPass(const glm::mat4& view, const glm::mat4& proj);
    void renderTransparentPass(Backend::Texture* shadowTexture);
    void renderDebugOverlayPass();

  public:
    Rasterizer(Backend::GraphicsDevice* graphicsDevice);

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

    // shadow
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

    void setFrustumCullingEnabled(bool enabled) {
        _frustumCullingEnabled = enabled;
    }
    bool isFrustumCullingEnabled() const { return _frustumCullingEnabled; }
    void setDebugRenderAABB(bool enabled) { _debugRenderAABB = enabled; }
    bool getDebugRenderAABB() const { return _debugRenderAABB; }
    int getCullingTotalBatches() const { return _cullingTotalBatches; }
    int getCullingCulledBatches() const { return _cullingCulledBatches; }
    int getCullingTotalInstances() const { return _cullingTotalInstances; }
    int getCullingCulledInstances() const { return _cullingCulledInstances; }

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
    void removePrim(RenderableHandle handle, Scene::Prim* prim);

    void
    updateRenderableTransforms(RenderableHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors = nullptr);

    void setRenderableColors(RenderableHandle handle,
                             const std::vector<glm::vec4>& colors);

    // Disable back-face culling for this instancer (e.g. cloth, thin surfaces).
    void setRenderableDoubleSided(RenderableHandle handle,
                                  bool doubleSided = true);
    void setRenderableCastsShadow(RenderableHandle handle,
                                  bool castsShadow = true);
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

    void setSkybox(const std::string& path, UpAxis upAxis = UpAxis::Y) {
        _graphicsDevice->setSkybox(path, upAxis);
    }
    void setSkybox(const std::vector<std::string>& paths,
                   UpAxis upAxis = UpAxis::Y) {
        _graphicsDevice->setSkybox(paths, upAxis);
    }

    void updateFrameData(const glm::mat4& view, const glm::mat4& proj);
    void render(const glm::mat4& view, const glm::mat4& proj) override;
    void renderSelectionMaskPass(const RayPickResult& selection,
                                 Backend::Framebuffer* target, int width,
                                 int height);
};

} // namespace KE

#endif

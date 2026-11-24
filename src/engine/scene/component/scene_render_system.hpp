#ifndef _SCENE_RENDER_SYSTEM_HPP_
#define _SCENE_RENDER_SYSTEM_HPP_

#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/graphics/renderer/render_backend/render_backend.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/render/render_scene_data_collector.hpp"

#include <cstddef>
#include <glm/mat4x4.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace KE {

class Material;
class Renderer;
class App;

namespace Backend { class Texture; }

namespace Scene {

class Prim;
class RenderComponent;
class DebugDraw;

// Registers, updates, and unregisters RenderComponents with the render backend.
// Tracks registered Prims and execution batches; the backend owns GPU resources.
class SceneRenderSystem {
  public:
    SceneRenderSystem() = default;
    ~SceneRenderSystem();

    SceneRenderSystem(const SceneRenderSystem&) = delete;
    SceneRenderSystem& operator=(const SceneRenderSystem&) = delete;

    void bind(Renderer* renderer, SceneResourceManager* resources = nullptr);
    void syncScene(SceneBackend& scene, const SceneResourceManager& resources);
    RayPickResult resolvePick(const RenderPickHit& hit) const;
    RayPickResult buildPrimSelection(Prim& prim) const;

    std::shared_ptr<RenderComponent>
    addRenderable(Prim& prim, Material* material,
                  TransformSource source = TransformSource::SceneGraph);
    std::shared_ptr<RenderComponent>
    addSkinnedRenderable(Prim& prim, Material* material,
                         const SkinnedMeshData& skinnedMesh,
                         TransformSource source = TransformSource::SceneGraph);

    bool unregister(RenderComponent& component);
    void detachSubtree(Prim& root);

    void setDoubleSided(RenderComponent& component, bool doubleSided);
    void setCastsShadow(RenderComponent& component, bool castsShadow);
    void setAlphaMode(RenderComponent& component, AlphaMode mode,
                      float cutoff = 0.5f);
    bool setMaterial(RenderComponent& component, Material* material);
    // Native ColorMaterial overrides apply to the entire execution batch.
    void setBatchTexture(RenderComponent& component, Backend::Texture* texture,
                         TextureRole role);
    void setBatchTexture(RenderComponent& component, Backend::Texture* texture,
                         int slot);
    // TODO: Remove these aliases after callers migrate to setBatchTexture().
    void setTexture(RenderComponent& component, Backend::Texture* texture,
                    TextureRole role) { setBatchTexture(component, texture, role); }
    void setTexture(RenderComponent& component, Backend::Texture* texture,
                    int slot) { setBatchTexture(component, texture, slot); }
    void setExternalBuffer(RenderComponent& component,
                           const ExternalBufferDesc& desc);
    void setInstanceColors(RenderComponent& component,
                           const std::vector<glm::vec4>& colors);
    void updateInstances(RenderComponent& component,
                         const std::vector<glm::mat4>& transforms,
                         const std::vector<glm::vec4>* colors = nullptr);
    void updateGeometry(RenderComponent& component,
                        const std::vector<glm::vec3>& positions,
                        const std::vector<glm::vec3>& normals = {});
    std::shared_ptr<const RenderGeometryData>
    geometry(const RenderComponent& component) const;
    void updateSkinning(RenderComponent& component,
                        const std::vector<glm::mat4>& boneMatrices);

    bool isRegistered(const RenderComponent& component) const;
    bool hasBatch(const RenderComponent& component) const;
    bool sharesBatch(const RenderComponent& first,
                     const RenderComponent& second) const;
    size_t registrationCount() const { return _registrations.size(); }

    // Execution batch handle; absent -> InvalidHandle, source-only -> error.
    RenderableHandle handle(const RenderComponent& component) const;

  private:
    friend class ::KE::App;
    friend class DebugDraw;

    struct Registration {
        std::weak_ptr<RenderComponent> component;
        Prim* prim = nullptr;
        // Source-only registrations have no execution batch handle.
        RenderableHandle handle = InvalidHandle;
        const MeshData* mesh = nullptr;
        std::shared_ptr<const RenderGeometryData> geometry;
        Material* material = nullptr;
        SceneBatchState state;
        std::optional<SkinnedMeshData> skinnedMesh;
        std::vector<std::pair<Backend::Texture*, int>> textureBindings;
    };

    void validateRegistration(
        const std::shared_ptr<RenderComponent>& component) const;
    void registerResources(RenderComponent& component, Material* material);
    bool registerRenderable(const std::shared_ptr<RenderComponent>& component,
                            Material* material);
    bool
    registerSkinnedRenderable(const std::shared_ptr<RenderComponent>& component,
                              Material* material,
                              const SkinnedMeshData& skinnedMesh);
    const Registration&
    requireRegistration(const RenderComponent& component) const;
    const Registration& requireBatch(const RenderComponent& component) const;
    void syncState(RenderComponent& component);
    void finishRegistration(const std::shared_ptr<RenderComponent>& component,
                            RenderableHandle handle, Material* material,
                            const SkinnedMeshData* skinnedMesh = nullptr);
    void clear();

    Renderer* _renderer = nullptr;
    SceneResourceManager* _resources = nullptr;
    RenderSceneDataCollector _collector;
    std::unordered_map<const RenderComponent*, Registration> _registrations;
};

} // namespace Scene
} // namespace KE

#endif

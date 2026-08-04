#ifndef _SCENE_RENDER_SYSTEM_HPP_
#define _SCENE_RENDER_SYSTEM_HPP_

#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/scene/scene_backend.hpp"

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

// Owns registration between scene RenderComponents and renderer batches.
// RenderableHandle remains private to this system during normal scene use.
class SceneRenderSystem {
  public:
    SceneRenderSystem() = default;
    ~SceneRenderSystem();

    SceneRenderSystem(const SceneRenderSystem&) = delete;
    SceneRenderSystem& operator=(const SceneRenderSystem&) = delete;

    void bind(Renderer* renderer);

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
    void setTexture(RenderComponent& component, Backend::Texture* texture,
                    TextureRole role);
    void setTexture(RenderComponent& component, Backend::Texture* texture,
                    int slot);
    void setExternalBuffer(RenderComponent& component,
                           const ExternalBufferDesc& desc);
    void updateInstances(RenderComponent& component,
                         const std::vector<glm::mat4>& transforms,
                         const std::vector<glm::vec4>* colors = nullptr);
    void updateGeometry(RenderComponent& component,
                        const std::vector<glm::vec3>& positions,
                        const std::vector<glm::vec3>& normals = {});
    void updateSkinning(RenderComponent& component,
                        const std::vector<glm::mat4>& boneMatrices);

    bool isRegistered(const RenderComponent& component) const;
    bool sharesBatch(const RenderComponent& first,
                     const RenderComponent& second) const;
    size_t registrationCount() const { return _registrations.size(); }

  private:
    friend class ::KE::App;
    friend class DebugDraw;

    struct Registration {
        std::weak_ptr<RenderComponent> component;
        Prim* prim = nullptr;
        RenderableHandle handle = InvalidHandle;
        std::optional<SkinnedMeshData> skinnedMesh;
        std::vector<std::pair<Backend::Texture*, int>> textureBindings;
    };

    void validateRegistration(
        const std::shared_ptr<RenderComponent>& component) const;
    RenderableHandle
    registerRenderable(const std::shared_ptr<RenderComponent>& component,
                       Material* material);
    RenderableHandle
    registerSkinnedRenderable(const std::shared_ptr<RenderComponent>& component,
                              Material* material,
                              const SkinnedMeshData& skinnedMesh);
    const Registration&
    requireRegistration(const RenderComponent& component) const;
    void syncState(RenderComponent& component);
    RenderableHandle
    finishRegistration(const std::shared_ptr<RenderComponent>& component,
                       RenderableHandle handle,
                       const SkinnedMeshData* skinnedMesh = nullptr);
    RenderableHandle handle(const RenderComponent& component) const;
    void clear();

    Renderer* _renderer = nullptr;
    std::unordered_map<const RenderComponent*, Registration> _registrations;
};

} // namespace Scene
} // namespace KE

#endif

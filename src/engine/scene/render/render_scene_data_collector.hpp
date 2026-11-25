#pragma once

// Collects scene inputs and tracks accepted updates for render backends.

#include "engine/scene/render/render_scene_data.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace KE {
namespace Scene {

class Prim;
class RenderComponent;
class SceneBackend;
class SceneResourceManager;
class SceneRenderSystem;
class TransformComponent;

// Caches membership; visibility and world poses are evaluated on each
// collection.
class RenderLightCollector {
  public:
    RenderLightData collect(Prim* root);

  private:
    std::weak_ptr<TransformComponent> _root;
    uint64_t _structureVersion = 0;
    std::vector<Prim*> _prims;
};

// Advances its baseline only after a backend accepts the submitted update.
class RenderSceneUpdateTracker {
  public:
    RenderSceneFrame prepare(const RenderSceneFrame& frame) const;
    void accept(const RenderSceneFrame& update);

  private:
    std::map<RenderInstanceId, RenderObjectSource> _objects;
    RenderLightData _lighting;
    bool _initialized = false;
};

// Evaluates scene state; cached Prim pointers are non-owning and structure-scoped.
class RenderSceneDataCollector {
  public:
    RenderSceneFrame collect(SceneBackend& scene,
                             const SceneResourceManager& resources,
                             const std::string& cameraPath);
    // App supplies its camera separately and submits only registered objects.
    RenderSceneFrame collectRegistered(SceneBackend& scene,
                                       const SceneResourceManager& resources,
                                       const SceneRenderSystem& registrations);
    std::shared_ptr<RenderComponent> resolve(RenderInstanceId id) const;
    RenderInstanceId instanceId(const Prim& prim) const;
    void forget(const RenderComponent& component);

  private:
    RenderLightCollector _lights;
    std::weak_ptr<TransformComponent> _root;
    uint64_t _structureVersion = 0;
    std::vector<Prim*> _prims;
    RenderSceneFrame collectScene(SceneBackend& scene,
                                  const SceneResourceManager& resources,
                                  const SceneRenderSystem* registrations);
    struct Identity {
        std::weak_ptr<RenderComponent> component;
        RenderInstanceId id;
    };
    std::map<std::string, Identity> _identities;
    RenderInstanceId _nextId = 1;
};

} // namespace Scene
} // namespace KE

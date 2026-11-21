#pragma once

// Collects scene and resource data for Native/ovrtx image rendering.

#include "engine/scene/render_scene_data.hpp"
#include "engine/scene/render_material_data.hpp"
#include "engine/scene/scene_resource_manager.hpp"
#include "engine/scene/render_light_data.hpp"

#include <map>
#include <string>

namespace KE {
namespace Scene {

class RenderComponent;
class SceneRenderSystem;

// One object's source resources and evaluated world state.
struct RenderObjectSource {
    RenderInstanceId id = InvalidRenderInstance;
    std::string path;
    RenderMeshData mesh;
    RenderMaterialData material;
    ResourceHandle materialResource = InvalidResourceHandle;
    RenderTextureSources textures;
    glm::mat4 world{1.0f};
    glm::vec4 color{1.0f};
    bool visible = true;
    bool pickable = true;
    bool doubleSided = false;
};

struct RenderCameraSource {
    glm::mat4 world{1.0f};
    float verticalFov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
};

// Full scene state or an incremental update consumed by either render adapter.
struct RenderSceneFrame {
    std::vector<RenderObjectSource> objects;
    RenderCameraSource camera;
    RenderLightData lighting;
    // Incremental frames upsert objects and remove only the listed identities.
    bool incremental = false;
    std::vector<RenderInstanceId> removedObjects;
    bool lightingChanged = true;
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

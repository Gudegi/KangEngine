///
/// PhysicsBridge — syncs PhysX simulation results (pose) to scene Prim visuals.
///
/// Usage:
///   PhysicsBridge bridge{this};               // App* once at construction
///   bridge.add(artic, skelBridge);            // single robot (Prim-based)
///   bridge.addInstanced(artics, handles);     // N robots   (Handle-based)
///   bridge.addInstanced(artic, handles);      // append one robot to group
///   bridge.addCollisionVisuals(artic, scene); // collision debug (optional)
///
///   bridge.sync();                            // once per frame
///

#ifndef _PHYSICS_BRIDGE_HPP_
#define _PHYSICS_BRIDGE_HPP_

#include "engine/graphics/renderer/rasterizer.hpp"
#include "PxPhysicsAPI.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <string>
#include <vector>

namespace KE {

class App;          // forward declaration
class Articulation; // forward declaration

namespace Bridge {
class SkeletonBridge; // forward declaration
} // namespace Bridge

namespace Scene {
class SceneBackend;
class Prim;
} // namespace Scene

namespace Bridge {

class PhysicsBridge {
  public:
    explicit PhysicsBridge(App* app = nullptr) : _app(app) {}

    // Single articulation paired with skeleton visuals — Prim-based sync
    void add(const Articulation& artic, const SkeletonBridge& skelBridge);

    // N identical articulations, one handle per body type — instanced sync
    void addInstanced(std::vector<Articulation*> artics,
                      std::vector<MeshHandle> handles);
    void addInstanced(const Articulation& artic,
                      const std::vector<MeshHandle>& handles);

    // Create one Prim per collision geom. Returns Prims for addShape().
    // visibleByDefault=false: debug overlay (invisible until toggled)
    std::vector<Scene::Prim*>
    addCollisionVisuals(const Articulation& artic, Scene::SceneBackend* scene,
                        const std::string& basePath = "/collision",
                        bool visibleByDefault = false);

    // Sync all registered visuals — call once per frame
    void sync();

    void setCollisionVisible(bool visible);

  private:
    App* _app = nullptr;

    struct PrimVisual {
        physx::PxArticulationLink* link;
        Scene::Prim* prim;
    };

    struct InstancedGroup {
        std::vector<const Articulation*> artics;
        std::vector<MeshHandle> handles;
    };

    struct ColVisual {
        physx::PxArticulationLink* link = nullptr;
        Scene::Prim* prim = nullptr;
        glm::vec3 localPos{0.f};
        glm::quat localQuat{1.f, 0.f, 0.f, 0.f};
    };

    std::vector<PrimVisual> _primVisuals;
    std::vector<InstancedGroup> _instancedGroups;
    std::vector<ColVisual> _colVisuals;
};

} // namespace Bridge
} // namespace KE

#endif

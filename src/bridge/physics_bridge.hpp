///
/// PhysicsBridge — adapter from PhysX state to scene/render visuals.
///
/// This class does not own simulation state. It only copies PhysX poses into
/// SceneGraph prims for small-scene compatibility and collision debugging.
///
/// Usage:
///   PhysicsBridge bridge;
///   bridge.add(artic, skelBridge);            // single robot (Prim-based)
///   bridge.addCollisionVisuals(artic, scene); // collision debug (optional)
///
///   bridge.sync();                            // once per frame
///

#ifndef _PHYSICS_BRIDGE_HPP_
#define _PHYSICS_BRIDGE_HPP_

#include "PxPhysicsAPI.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <string>
#include <vector>

namespace KE {

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
    PhysicsBridge() = default;

    // Small-scene compatibility path. Batched simulation visuals use
    // SimVisualBatch and ExternalBuffer directly.
    void add(const Articulation& artic, const SkeletonBridge& skelBridge);

    // Create one Prim per collision geom. Returns Prims for addRenderable().
    // visibleByDefault=false: debug overlay (invisible until toggled)
    // Debug visual authoring helper, not the primary simulation sync path.
    std::vector<Scene::Prim*>
    addCollisionVisuals(const Articulation& artic, Scene::SceneBackend* scene,
                        const std::string& basePath = "/collision",
                        bool visibleByDefault = false);

    // Sync all registered visuals — call once per frame
    void sync();

    void setCollisionVisible(bool visible);

  private:
    struct PrimVisual {
        physx::PxArticulationLink* link;
        Scene::Prim* prim;
    };

    struct ColVisual {
        physx::PxArticulationLink* link = nullptr;
        Scene::Prim* prim = nullptr;
        glm::vec3 localPos{0.f};
        glm::quat localQuat{1.f, 0.f, 0.f, 0.f};
    };

    std::vector<PrimVisual> _primVisuals;
    std::vector<ColVisual> _colVisuals;
};

} // namespace Bridge
} // namespace KE

#endif

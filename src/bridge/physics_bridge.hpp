///
/// PhysicsBridge — syncs PhysX simulation results (pose) to scene Prim visuals.
/// Also owns collision shape visualization (build + sync + visibility).
///

#ifndef _PHYSICS_BRIDGE_HPP_
#define _PHYSICS_BRIDGE_HPP_

#include "PxPhysicsAPI.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

using namespace physx;

namespace KE {

class Articulation; // forward — defined in physics/articulation.hpp

namespace Bridge {
class SkeletonBridge; // forward — defined in bridge/skeleton_bridge.hpp
} // namespace Bridge

namespace Scene {
class SceneBackend;
class Prim;
} // namespace Scene

namespace Bridge {

class PhysicsBridge {
  public:
    // --- Rigid body sync (skeleton FK visuals) ---
    struct RigidVisual {
        PxArticulationLink* link = nullptr;
        PxRigidDynamic* rigid = nullptr;
        Scene::Prim* prim = nullptr;
    };

    void registerRigidVisual(const RigidVisual& rv) {
        _rigidVisuals.push_back(rv);
    }

    // Read PhysX poses → write to each Prim's xformOp attributes.
    void syncAllVisuals();

    // --- Collision shape visualization ---
    struct ColVisual {
        PxArticulationLink* link = nullptr;
        Scene::Prim* prim = nullptr;
        glm::vec3 localPos{0.f};
        glm::quat localQuat{1.f, 0.f, 0.f, 0.f};
    };

    // Create one Prim per collision geom from artic. Prims are invisible by
    // default. Returns created prims so the caller can addShape(shader, prim).
    std::vector<Scene::Prim*>
    buildCollisionVisuals(const Articulation& artic, Scene::SceneBackend* scene,
                          const std::string& basePath = "/collision");

    // Sync collision Prim positions from PhysX link poses.
    void syncCollisionVisuals();

    void setCollisionVisible(bool visible);

    // Register one RigidVisual per articulation link paired with the
    // corresponding SkeletonBridge body Prim.
    void registerArticulationVisuals(const Articulation& artic,
                                     const SkeletonBridge& skelBridge);

  private:
    std::vector<RigidVisual> _rigidVisuals;
    std::vector<ColVisual> _colVisuals;
};

} // namespace Bridge
} // namespace KE

#endif

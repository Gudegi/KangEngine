#ifndef _ARTICULATION_HPP_
#define _ARTICULATION_HPP_

#include "animation/mjcf_loader.hpp"
#include "animation/robot.hpp"
#include "physics.hpp"
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

namespace KE::Scene {
class SceneBackend;
class Prim;
} // namespace KE::Scene

namespace KE {

struct ArticulationConfig {
    bool fixBase = true;
    bool disableSelfCollision = true;
    int solverIterations = 16;

    float defaultRootMass = 8.f;
    float defaultLinkMass = 1.5f;

    // PhysX defaults: linearDamping=0, angularDamping=0.05, maxAngVel=100
    float rootLinearDamping = 0.f;
    float rootAngularDamping = 0.05f;
    float linkLinearDamping = 0.f;
    float linkAngularDamping = 0.05f;
    float maxAngularVelocity = 100.f;
    bool enableCCD = false;

    // Fallback shapes when collisionGeoms has no entry for a body
    PxVec3 rootBoxHalf = {0.15f, 0.1f, 0.1f};
    PxVec3 linkBoxHalf = {0.04f, 0.04f, 0.04f};

    static ArticulationConfig fixedBase() {
        return {}; // all defaults
    }

    static ArticulationConfig freeBase() {
        ArticulationConfig config;
        config.fixBase = false;
        config.disableSelfCollision = true;
        config.solverIterations = 32;
        config.rootLinearDamping = 0.02f;
        config.rootAngularDamping = 0.1f;
        config.linkLinearDamping = 0.1f;
        config.linkAngularDamping = 0.5f;
        config.maxAngularVelocity = 25.f;
        config.enableCCD = true;
        return config;
    }
};

struct ColVisual {
    PxArticulationLink* link = nullptr;
    Scene::Prim* prim = nullptr;
    glm::vec3 localPos{0.f};
    glm::quat localQuat{1.f, 0.f, 0.f, 0.f};
};

class Articulation {
  private:
    PxArticulationReducedCoordinate* _artic = nullptr;
    std::vector<PxArticulationLink*> _links;
    std::vector<Animation::Joint> _joints; // indexed by body idx (root unused)
    Animation::CollisionGeomMap _colGeoms;
    std::vector<ColVisual> _colVisuals;

  public:
    Articulation() = default;
    ~Articulation();

    Articulation(const Articulation&) = delete;
    Articulation& operator=(const Articulation&) = delete;
    Articulation(Articulation&&) noexcept;
    Articulation& operator=(Articulation&&) noexcept;

    static Articulation build(PhysicsWorld& physics,
                              Animation::SkelMesh& skelMesh,
                              const std::string& mjcfPath,
                              const ArticulationConfig& cfg = {});

    void setDriveTargets(const std::vector<float>& targets, float kp, float kd);
    void resetRoot(const PxTransform& pose);

    // Collision visualization — call after build()
    // Creates one Prim per collision geom (invisible by default).
    // Returns the created prims so the caller can addShape(shader, prim).
    std::vector<Scene::Prim*>
    buildCollisionVisuals(Scene::SceneBackend* scene,
                          const std::string& basePath = "/collision");
    void syncCollisionVisuals();
    void setCollisionVisible(bool visible);

    PxArticulationLink* link(int i) const { return _links[i]; }
    int numLinks() const { return static_cast<int>(_links.size()); }
    PxArticulationReducedCoordinate* raw() { return _artic; }
    const std::vector<Animation::Joint>& joints() const { return _joints; }
};

} // namespace KE

#endif

#ifndef _ARTICULATION_HPP_
#define _ARTICULATION_HPP_

#include "animation/character_description.hpp"
#include "physics.hpp"
#include <memory>
#include <string>
#include <vector>

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

class Articulation {
  private:
    PxArticulationReducedCoordinate* _artic = nullptr;
    std::vector<PxArticulationLink*> _links;
    Animation::JointMap _joints; // body index -> joints (empty = fixed)
    Animation::CollisionGeomMap _colGeoms;

  public:
    Articulation() = default;
    ~Articulation();

    Articulation(const Articulation&) = delete;
    Articulation& operator=(const Articulation&) = delete;
    Articulation(Articulation&&) noexcept;
    Articulation& operator=(Articulation&&) noexcept;

    static Articulation
    build(PhysicsWorld& physics,
          std::shared_ptr<const Animation::SkeletonTree> tree,
          const Animation::CollisionGeomMap& colGeoms,
          const Animation::JointMap& joints,
          const Animation::InertialMap& inertials,
          const ArticulationConfig& cfg = {});

    void setDriveTargets(const std::vector<float>& targets, float kp, float kd);
    void resetRoot(const PxTransform& pose);

    PxArticulationLink* link(int i) const { return _links[i]; }
    int numLinks() const { return static_cast<int>(_links.size()); }
    PxArticulationReducedCoordinate* raw() { return _artic; }
    const Animation::JointMap& joints() const { return _joints; }
    int numDofs() const {
        int total = 0;
        for (const auto& kv : _joints)
            total += static_cast<int>(kv.second.size());
        return total;
    }

    // Data accessors for PhysicsBridge
    const std::vector<PxArticulationLink*>& links() const { return _links; }
    const Animation::CollisionGeomMap& colGeoms() const { return _colGeoms; }
};

} // namespace KE

#endif

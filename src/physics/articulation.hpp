#ifndef _ARTICULATION_HPP_
#define _ARTICULATION_HPP_

#include "animation/character_description.hpp"
#include "physics.hpp"
#include <array>
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
    PxU32 collisionGroup = 0;
    float contactOffset = 0.02f;
    float restOffset = 0.f;

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
    struct DofInfo {
        int linkIndex = -1;
        std::string name;
        PxArticulationAxis::Enum axis = PxArticulationAxis::eTWIST;
        float loLimit = 0.f;
        float hiLimit = 0.f;
        float kp = 0.f;
        float kd = 0.f;
        float effortLimit = PX_MAX_F32;
    };

    PxArticulationReducedCoordinate* _artic = nullptr;
    std::vector<PxArticulationLink*> _links;
    Animation::JointMap _joints; // body index -> joints (empty = fixed)
    Animation::CollisionGeomMap _colGeoms;
    std::vector<DofInfo> _dofs;
    std::vector<float> _KPs;
    std::vector<float> _KDs;
    std::vector<float> _effortLimits;
    std::vector<float> _appliedForces;

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

    void release();
    void setDriveTargets(const std::vector<float>& targets, float kp, float kd);
    void setDriveTargets(const std::vector<float>& targets);
    void setDriveVelocityTargets(const std::vector<float>& targets);
    void resetRoot(const PxTransform& pose);
    void setRootState(const PxTransform& pose, const PxVec3& linearVelocity,
                      const PxVec3& angularVelocity);
    void setDofState(const std::vector<float>& positions,
                     const std::vector<float>& velocities);
    void setJointForces(const std::vector<float>& forces);
    void addLinkForce(int linkIndex, const PxVec3& force);
    void setKPs(const std::vector<float>& kps);
    const std::vector<float>& getKPs() const { return _KPs; }
    void setKDs(const std::vector<float>& kds);
    const std::vector<float>& getKDs() const { return _KDs; }
    void setEffortLimits(const std::vector<float>& effortLimits);
    const std::vector<float>& getEffortLimits() const { return _effortLimits; }

    PxArticulationLink* link(int i) const { return _links[i]; }
    int numLinks() const { return static_cast<int>(_links.size()); }
    PxArticulationReducedCoordinate* raw() { return _artic; }
    const Animation::JointMap& joints() const { return _joints; }
    int numDofs() const { return static_cast<int>(_dofs.size()); }

    // Data accessors for PhysicsBridge
    const std::vector<PxArticulationLink*>& links() const { return _links; }
    const Animation::CollisionGeomMap& colGeoms() const { return _colGeoms; }

    // State queries for Python/Model-State integration.
    // Flat arrays use xyz for vectors and xyzw for quaternions.
    std::vector<float> getRootPositionFlat() const;
    std::vector<float> getRootRotationFlat() const;
    std::vector<float> getRootLinearVelocityFlat() const;
    std::vector<float> getRootAngularVelocityFlat() const;
    std::vector<float> getLinkPositionsFlat() const;
    std::vector<float> getLinkRotationsFlat() const;
    std::vector<float> getLinkLinearVelocitiesFlat() const;
    std::vector<float> getLinkAngularVelocitiesFlat() const;
    std::vector<float> getDofPositions() const;
    std::vector<float> getDofVelocities() const;
    std::vector<float> getDofForces() const;
    std::vector<std::string> getDofNames() const;
    std::vector<std::array<float, 2>> getDofLimits() const;
    std::vector<float> getDofEffortLimits() const;
    std::vector<float> getLinkMasses() const;
    float calcMass() const;
};

} // namespace KE

#endif

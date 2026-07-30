#ifndef _ARTICULATION_HPP_
#define _ARTICULATION_HPP_

#include "character/character_description.hpp"
#include "physics.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace KE {

struct ArticulationConfig {
    bool fixBase = true;
    bool disableSelfCollision = true;
    bool useAggregate = false;
    int solverPositionIterations = 16;
    int solverVelocityIterations = 1;

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

    // Fallback shapes for bodies whose MJCF authored a collidable mesh geom
    // that KangEngine cannot cook yet. Bodies with only visual-only geoms do
    // not receive fallback collision.
    PxVec3 rootBoxHalf = {0.075f, 0.075f, 0.075f};
    PxVec3 linkBoxHalf = {0.05f, 0.05f, 0.05f};

    // Build-time material overrides. Later entries win, so callers can set a
    // body-wide override and then refine one named geom.
    std::vector<Physics::CollisionMaterialOverride> materialOverrides;

    static ArticulationConfig fixedBase() {
        return {}; // all defaults
    }

    static ArticulationConfig freeBase() {
        ArticulationConfig config;
        config.fixBase = false;
        config.disableSelfCollision = true;
        config.solverPositionIterations = 32;
        config.solverVelocityIterations = 1;
        config.rootLinearDamping = 0.02f;
        config.rootAngularDamping = 0.1f;
        config.linkLinearDamping = 0.1f;
        config.linkAngularDamping = 0.5f;
        config.maxAngularVelocity = 25.f;
        config.enableCCD = true;
        return config;
    }
};

class ArticulationTemplate {
  public:
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

  private:
    std::shared_ptr<const Animation::SkeletonTree> _tree;
    Character::JointDescMap _joints;
    Character::CollisionGeomDescMap _colGeoms;
    Character::InertialDescMap _inertials;
    std::vector<PxTransform> _restTransforms;
    // Orientation of each inbound joint frame. Multi-axis MJCF joints need a
    // shared frame whose PhysX twist/swing axes match the authored axes.
    std::vector<PxQuat> _jointFrames;
    std::vector<std::string> _bodyNames;
    std::vector<DofInfo> _dofs;

    friend class Articulation;

  public:
    static std::shared_ptr<ArticulationTemplate>
    create(std::shared_ptr<const Animation::SkeletonTree> tree,
           const Character::CollisionGeomDescMap& colGeoms,
           const Character::JointDescMap& joints,
           const Character::InertialDescMap& inertials,
           const ArticulationConfig& cfg = {});

    int numLinks() const { return static_cast<int>(_bodyNames.size()); }
    int numDofs() const { return static_cast<int>(_dofs.size()); }
    const std::vector<std::string>& bodyNames() const { return _bodyNames; }
};

class Articulation {
  private:
    PxArticulationReducedCoordinate* _artic = nullptr;
    PxAggregate* _aggregate = nullptr;
    std::vector<PxArticulationLink*> _links;
    std::shared_ptr<ArticulationTemplate> _template;
    std::vector<float> _KPs;
    std::vector<float> _KDs;
    std::vector<float> _effortLimits;
    std::vector<float> _appliedForces;

    void syncDriveParams();
    std::vector<int> getDofPhysxIndices() const;

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
          const Character::CollisionGeomDescMap& colGeoms,
          const Character::JointDescMap& joints,
          const Character::InertialDescMap& inertials,
          const ArticulationConfig& cfg = {});
    static Articulation
    build(PhysicsWorld& physics,
          std::shared_ptr<ArticulationTemplate> articulationTemplate,
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
    void addLinkForceAtPosition(int linkIndex, const PxVec3& force,
                                const PxVec3& position);
    int setCollisionMaterial(PhysicsWorld& physics,
                             const Physics::PhysicsMaterialDesc& material);
    int setCollisionMaterialOverrides(
        PhysicsWorld& physics,
        const std::vector<Physics::CollisionMaterialOverride>& overrides);
    void setKPs(const std::vector<float>& kps);
    const std::vector<float>& getKPs() const { return _KPs; }
    void setKDs(const std::vector<float>& kds);
    const std::vector<float>& getKDs() const { return _KDs; }
    void setEffortLimits(const std::vector<float>& effortLimits);
    const std::vector<float>& getEffortLimits() const { return _effortLimits; }

    PxArticulationLink* link(int i) const { return _links[i]; }
    int numLinks() const { return static_cast<int>(_links.size()); }
    PxArticulationReducedCoordinate* raw() { return _artic; }
    const Character::JointDescMap& joints() const {
        static const Character::JointDescMap empty;
        return _template ? _template->_joints : empty;
    }
    int numDofs() const { return _template ? _template->numDofs() : 0; }
    std::shared_ptr<ArticulationTemplate> articulationTemplate() const {
        return _template;
    }

    // Data accessors for PhysicsBridge
    const std::vector<PxArticulationLink*>& links() const { return _links; }
    const Character::CollisionGeomDescMap& colGeoms() const {
        static const Character::CollisionGeomDescMap empty;
        return _template ? _template->_colGeoms : empty;
    }
    const std::vector<std::string>& bodyNames() const {
        static const std::vector<std::string> empty;
        return _template ? _template->_bodyNames : empty;
    }
    const std::string& bodyName(int index) const {
        return _template->_bodyNames[index];
    }

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
    std::vector<int> getLinkIndices() const;
    std::vector<float> getDofPositions() const;
    std::vector<float> getDofVelocities() const;
    std::vector<float> getDofForces() const;
    std::vector<std::string> getDofNames() const;
    std::vector<int> getDofGpuIndices() const;
    std::vector<std::array<float, 2>> getDofLimits() const;
    std::vector<float> getDofEffortLimits() const;
    std::vector<float> getLinkMasses() const;
    float calcMass() const;
};

} // namespace KE

#endif

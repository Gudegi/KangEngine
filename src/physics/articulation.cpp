#include "articulation.hpp"
#include "animation/skeleton_state.hpp"

#include <Eigen/Geometry>
#include <utility>

namespace KE {
namespace {

// Returns a PhysX quaternion that rotates UnitX onto the given axis.
PxQuat axisAlignQuat(Eigen::Vector3f axis) {
    axis.normalize();
    Eigen::Quaternionf q =
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), axis);
    return PxQuat(q.x(), q.y(), q.z(), q.w());
}

// Returns a PxTransform centered between from/to with X aligned to (to - from).
PxTransform fromToPose(const Eigen::Vector3f& from, const Eigen::Vector3f& to) {
    Eigen::Vector3f mid = (from + to) * 0.5f;
    Eigen::Vector3f dir = to - from;
    float len = dir.norm();
    if (len < 1e-6f)
        return PxTransform(PxVec3(mid.x(), mid.y(), mid.z()));
    Eigen::Quaternionf q =
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), dir / len);
    return PxTransform(PxVec3(mid.x(), mid.y(), mid.z()),
                       PxQuat(q.x(), q.y(), q.z(), q.w()));
}

// Converts MJCF shape orientation (Z-axis capsule/cylinder) to PhysX X-axis
// convention.
PxQuat mjcfShapeRot(const Eigen::Quaternionf& mjcfQuat) {
    Eigen::Vector3f axis = mjcfQuat * Eigen::Vector3f::UnitZ();
    Eigen::Quaternionf q =
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), axis);
    return PxQuat(q.x(), q.y(), q.z(), q.w());
}

// Creates and attaches PhysX shapes for each MJCF collision geom on the link.
// Cylinders are approximated as capsules (PhysX has no native cylinder shape).
void attachCollisionShapes(PxArticulationLink* link, PxMaterial* mat,
                           const Animation::CollisionGeom* geoms,
                           std::size_t count) {
    using Type = Animation::CollisionGeom::Type;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& g = geoms[i];
        PxShape* shape = nullptr;
        PxTransform localPose(PxIdentity);
        switch (g.type) {
        case Type::Capsule:
        case Type::Cylinder: {
            float radius = g.size[0];
            if (g.hasFromTo) {
                float halfH = (g.to - g.from).norm() * 0.5f;
                shape = PxRigidActorExt::createExclusiveShape(
                    *link, PxCapsuleGeometry(radius, halfH), *mat);
                localPose = fromToPose(g.from, g.to);
            } else {
                shape = PxRigidActorExt::createExclusiveShape(
                    *link, PxCapsuleGeometry(radius, g.size[1]), *mat);
                localPose = PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                                        mjcfShapeRot(g.quat));
            }
            break;
        }
        case Type::Sphere:
            shape = PxRigidActorExt::createExclusiveShape(
                *link, PxSphereGeometry(g.size[0]), *mat);
            localPose = PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()));
            break;
        case Type::Box:
            shape = PxRigidActorExt::createExclusiveShape(
                *link, PxBoxGeometry(g.size[0], g.size[1], g.size[2]), *mat);
            localPose = PxTransform(
                PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                PxQuat(g.quat.x(), g.quat.y(), g.quat.z(), g.quat.w()));
            break;
        }
        if (shape)
            shape->setLocalPose(localPose);
    }
}

// Convenience overload accepting a vector of geoms.
void attachCollisionShapes(PxArticulationLink* link, PxMaterial* mat,
                           const std::vector<Animation::CollisionGeom>& geoms) {
    attachCollisionShapes(link, mat, geoms.data(), geoms.size());
}

// Applies MJCF inertial properties (mass, COM, diag inertia) to a link.
// Falls back to uniform mass distribution if the link has no inertial entry.
void applyInertial(PxArticulationLink* link,
                   const Animation::InertialMap& inertials, int idx,
                   float fallbackMass = 1.f) {
    auto it = inertials.find(idx);
    if (it == inertials.end()) {
        PxRigidBodyExt::updateMassAndInertia(*link, fallbackMass);
        return;
    }
    const auto& inert = it->second;
    link->setMass(inert.mass);
    link->setCMassLocalPose(
        PxTransform(PxVec3(inert.com.x(), inert.com.y(), inert.com.z()),
                    PxQuat(inert.quat.x(), inert.quat.y(), inert.quat.z(),
                           inert.quat.w())));
    link->setMassSpaceInertiaTensor(PxVec3(
        inert.diagInertia.x(), inert.diagInertia.y(), inert.diagInertia.z()));
}

} // anonymous namespace

// Move semantics
Articulation::Articulation(Articulation&& o) noexcept
    : _artic(o._artic), _links(std::move(o._links)),
      _joints(std::move(o._joints)), _colGeoms(std::move(o._colGeoms)) {
    o._artic = nullptr;
}

Articulation& Articulation::operator=(Articulation&& o) noexcept {
    if (this != &o) {
        release();
        _artic = o._artic;
        _links = std::move(o._links);
        _joints = std::move(o._joints);
        _colGeoms = std::move(o._colGeoms);
        o._artic = nullptr;
    }
    return *this;
}

Articulation::~Articulation() { release(); }

void Articulation::release() {
    if (_artic) {
        _artic->release();
        _artic = nullptr;
    }
    _links.clear();
    _joints.clear();
    _colGeoms.clear();
}

Articulation Articulation::build(
    PhysicsWorld& physics, std::shared_ptr<const Animation::SkeletonTree> tree,
    const Animation::CollisionGeomMap& colGeoms,
    const Animation::JointMap& joints, const Animation::InertialMap& inertials,
    const ArticulationConfig& cfg) {
    // Compute rest-pose global transforms — no copy, shared_ptr passed
    // directly.
    auto state = Animation::SkeletonState::zeroPose(tree);
    const auto globals = state.computeGlobalTransforms();

    Articulation artic;
    artic._joints = joints;
    artic._colGeoms = colGeoms;
    int n = tree->numJoints();

    artic._artic = physics.getPhysics()->createArticulationReducedCoordinate();
    artic._artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE,
                                      cfg.fixBase);
    artic._artic->setArticulationFlag(
        PxArticulationFlag::eDISABLE_SELF_COLLISION, cfg.disableSelfCollision);
    artic._artic->setSolverIterationCounts(cfg.solverIterations);
    artic._links.resize(n, nullptr);

    // Root link
    auto& g0 = globals[0];
    artic._links[0] = artic._artic->createLink(
        nullptr, PxTransform(PxVec3(g0.translation.x(), g0.translation.y(),
                                    g0.translation.z())));
    {
        auto it = colGeoms.find(0);
        if (it != colGeoms.end() && !it->second.empty())
            attachCollisionShapes(artic._links[0], physics.getMaterial(),
                                  it->second);
        else
            PxRigidActorExt::createExclusiveShape(
                *artic._links[0],
                PxBoxGeometry(cfg.rootBoxHalf.x, cfg.rootBoxHalf.y,
                              cfg.rootBoxHalf.z),
                *physics.getMaterial());
    }
    applyInertial(artic._links[0], inertials, 0, cfg.defaultRootMass);
    artic._links[0]->setLinearDamping(cfg.rootLinearDamping);
    artic._links[0]->setAngularDamping(cfg.rootAngularDamping);
    if (cfg.enableCCD)
        artic._links[0]->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);

    // Child links
    for (int i = 1; i < n; i++) {
        int pi = tree->parentIndex(i);
        PxTransform childWorld(PxVec3(globals[i].translation.x(),
                                      globals[i].translation.y(),
                                      globals[i].translation.z()));
        PxTransform parentWorld(PxVec3(globals[pi].translation.x(),
                                       globals[pi].translation.y(),
                                       globals[pi].translation.z()));

        artic._links[i] =
            artic._artic->createLink(artic._links[pi], childWorld);
        {
            auto it = colGeoms.find(i);
            if (it != colGeoms.end() && !it->second.empty())
                attachCollisionShapes(artic._links[i], physics.getMaterial(),
                                      it->second);
            else
                PxRigidActorExt::createExclusiveShape(
                    *artic._links[i],
                    PxBoxGeometry(cfg.linkBoxHalf.x, cfg.linkBoxHalf.y,
                                  cfg.linkBoxHalf.z),
                    *physics.getMaterial());
        }
        applyInertial(artic._links[i], inertials, i, cfg.defaultLinkMass);
        artic._links[i]->setLinearDamping(cfg.linkLinearDamping);
        artic._links[i]->setAngularDamping(cfg.linkAngularDamping);
        if (cfg.maxAngularVelocity > 0.f)
            artic._links[i]->setMaxAngularVelocity(cfg.maxAngularVelocity);

        auto* joint = static_cast<PxArticulationJointReducedCoordinate*>(
            artic._links[i]->getInboundJoint());

        auto jit = joints.find(i);
        int ndof = (jit != joints.end()) ? (int)jit->second.size() : 0;

        if (ndof == 0) {
            joint->setJointType(PxArticulationJointType::eFIX);
            PxTransform childPose(PxIdentity);
            PxTransform parentPose = parentWorld.getInverse() * childWorld;
            joint->setParentPose(parentPose);
            joint->setChildPose(childPose);
        } else if (ndof == 1) {
            const auto& jd = jit->second[0];
            joint->setJointType(PxArticulationJointType::eREVOLUTE);
            PxQuat aq = axisAlignQuat(jd.axis);
            PxTransform childPose(PxVec3(0.f), aq);
            PxTransform parentPose =
                parentWorld.getInverse() * childWorld * childPose;
            joint->setParentPose(parentPose);
            joint->setChildPose(childPose);
            joint->setMotion(PxArticulationAxis::eTWIST,
                             PxArticulationMotion::eLIMITED);
            joint->setLimit(PxArticulationAxis::eTWIST, jd.loLimit, jd.hiLimit);
        } else {
            // Multiple DOF -> spherical; assumes body-aligned x/y/z axes
            joint->setJointType(PxArticulationJointType::eSPHERICAL);
            PxTransform childPose(PxIdentity);
            PxTransform parentPose = parentWorld.getInverse() * childWorld;
            joint->setParentPose(parentPose);
            joint->setChildPose(childPose);
            for (const auto& jd : jit->second) {
                PxArticulationAxis::Enum axis;
                if (jd.axis.isApprox(Eigen::Vector3f::UnitX(), 0.01f))
                    axis = PxArticulationAxis::eTWIST;
                else if (jd.axis.isApprox(Eigen::Vector3f::UnitY(), 0.01f))
                    axis = PxArticulationAxis::eSWING2;
                else
                    axis = PxArticulationAxis::eSWING1;
                joint->setMotion(axis, PxArticulationMotion::eLIMITED);
                joint->setLimit(axis, jd.loLimit, jd.hiLimit);
            }
        }
    }

    physics.getScene()->addArticulation(*artic._artic);
    return artic;
}

// Drive
void Articulation::setDriveTargets(const std::vector<float>& targets, float kp,
                                   float kd) {
    PxArticulationDrive drive{kp, kd, PX_MAX_F32,
                              PxArticulationDriveType::eFORCE};

    int n = static_cast<int>(_links.size());
    int dofIdx = 0;
    for (int i = 1; i < n; i++) {
        auto jit = _joints.find(i);
        if (jit == _joints.end() || jit->second.empty())
            continue;

        auto* joint = static_cast<PxArticulationJointReducedCoordinate*>(
            _links[i]->getInboundJoint());

        for (const auto& jd : jit->second) {
            if (dofIdx >= static_cast<int>(targets.size()))
                break;
            PxArticulationAxis::Enum axis;
            if (jit->second.size() == 1) {
                axis = PxArticulationAxis::eTWIST;
            } else if (jd.axis.isApprox(Eigen::Vector3f::UnitX(), 0.01f)) {
                axis = PxArticulationAxis::eTWIST;
            } else if (jd.axis.isApprox(Eigen::Vector3f::UnitY(), 0.01f)) {
                axis = PxArticulationAxis::eSWING2;
            } else {
                axis = PxArticulationAxis::eSWING1;
            }
            joint->setDriveParams(axis, drive);
            joint->setDriveTarget(axis, targets[dofIdx++]);
        }
    }

    if (kp > 0.f && _artic->isSleeping())
        _artic->wakeUp();
}

// Reset root
void Articulation::resetRoot(const PxTransform& pose) {
    PxArticulationCache* cache = _artic->createCache();
    _artic->copyInternalStateToCache(*cache, PxArticulationCacheFlag::eALL);
    _artic->zeroCache(*cache);
    if (cache->rootLinkData) {
        cache->rootLinkData->transform = pose;
        cache->rootLinkData->worldLinVel = PxVec3(0.f);
        cache->rootLinkData->worldAngVel = PxVec3(0.f);
    }
    _artic->applyCache(*cache, PxArticulationCacheFlag::eALL);
    cache->release();
}

} // namespace KE

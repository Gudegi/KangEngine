#include "articulation.hpp"
#include "animation/skeleton_state.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace KE {
namespace {

PxQuat toPxQuat(const Eigen::Quaternionf& q) {
    return PxQuat(q.x(), q.y(), q.z(), q.w());
}

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

void applyContactOffsets(PxShape* shape, float contactOffset,
                         float restOffset) {
    if (!shape)
        return;
    shape->setRestOffset(restOffset);
    shape->setContactOffset(std::max(contactOffset, restOffset + 1e-4f));
}

// Creates and attaches PhysX shapes for each MJCF collision geom on the link.
// Cylinders are approximated as capsules (PhysX has no native cylinder shape).
void attachCollisionShapes(PxArticulationLink* link, PxPhysics* physics,
                           PxMaterial* mat,
                           const Animation::CollisionGeom* geoms,
                           std::size_t count, float contactOffset,
                           float restOffset) {
    using Type = Animation::CollisionGeom::Type;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& g = geoms[i];
        PxMaterial* shapeMat = mat;
        PxMaterial* ownedMat = nullptr;
        const float friction = g.friction;
        if (physics && std::abs(friction - 1.f) > 1e-6f) {
            ownedMat = physics->createMaterial(friction, friction, 0.f);
            ownedMat->setFrictionCombineMode(PxCombineMode::eMIN);
            shapeMat = ownedMat;
        }
        PxShape* shape = nullptr;
        PxTransform localPose(PxIdentity);
        switch (g.type) {
        case Type::Capsule:
        case Type::Cylinder: {
            float radius = g.size[0];
            if (g.hasFromTo) {
                float halfH = (g.to - g.from).norm() * 0.5f;
                shape = PxRigidActorExt::createExclusiveShape(
                    *link, PxCapsuleGeometry(radius, halfH), *shapeMat);
                localPose = fromToPose(g.from, g.to);
            } else {
                shape = PxRigidActorExt::createExclusiveShape(
                    *link, PxCapsuleGeometry(radius, g.size[1]), *shapeMat);
                localPose = PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                                        mjcfShapeRot(g.quat));
            }
            break;
        }
        case Type::Sphere:
            shape = PxRigidActorExt::createExclusiveShape(
                *link, PxSphereGeometry(g.size[0]), *shapeMat);
            localPose = PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()));
            break;
        case Type::Box:
            shape = PxRigidActorExt::createExclusiveShape(
                *link, PxBoxGeometry(g.size[0], g.size[1], g.size[2]),
                *shapeMat);
            localPose = PxTransform(
                PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                PxQuat(g.quat.x(), g.quat.y(), g.quat.z(), g.quat.w()));
            break;
        }
        if (shape) {
            shape->setLocalPose(localPose);
            applyContactOffsets(
                shape, g.margin >= 0.f ? g.margin : contactOffset, restOffset);
        }
        if (ownedMat)
            ownedMat->release();
    }
}

// Convenience overload accepting a vector of geoms.
void attachCollisionShapes(PxArticulationLink* link, PxPhysics* physics,
                           PxMaterial* mat,
                           const std::vector<Animation::CollisionGeom>& geoms,
                           float contactOffset, float restOffset) {
    attachCollisionShapes(link, physics, mat, geoms.data(), geoms.size(),
                          contactOffset, restOffset);
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

PxArticulationJointReducedCoordinate*
inboundJoint(const std::vector<PxArticulationLink*>& links, int linkIndex) {
    if (linkIndex <= 0 || linkIndex >= static_cast<int>(links.size()) ||
        !links[linkIndex])
        return nullptr;
    return static_cast<PxArticulationJointReducedCoordinate*>(
        links[linkIndex]->getInboundJoint());
}

void setCollisionFilterData(PxRigidActor* actor, PxU32 collisionGroup) {
    if (!actor || collisionGroup == 0)
        return;

    PxFilterData filterData;
    filterData.word2 = collisionGroup;
    filterData.word3 = 1;

    const PxU32 numShapes = actor->getNbShapes();
    std::vector<PxShape*> shapes(numShapes);
    actor->getShapes(shapes.data(), numShapes);
    for (PxShape* shape : shapes) {
        if (!shape)
            continue;
        shape->setSimulationFilterData(filterData);
        shape->setQueryFilterData(filterData);
    }
}

} // anonymous namespace

// Move semantics
Articulation::Articulation(Articulation&& o) noexcept
    : _artic(o._artic), _links(std::move(o._links)),
      _joints(std::move(o._joints)), _colGeoms(std::move(o._colGeoms)),
      _dofs(std::move(o._dofs)), _KPs(std::move(o._KPs)),
      _KDs(std::move(o._KDs)), _effortLimits(std::move(o._effortLimits)),
      _appliedForces(std::move(o._appliedForces)) {
    o._artic = nullptr;
}

Articulation& Articulation::operator=(Articulation&& o) noexcept {
    if (this != &o) {
        release();
        _artic = o._artic;
        _links = std::move(o._links);
        _joints = std::move(o._joints);
        _colGeoms = std::move(o._colGeoms);
        _dofs = std::move(o._dofs);
        _KPs = std::move(o._KPs);
        _KDs = std::move(o._KDs);
        _effortLimits = std::move(o._effortLimits);
        _appliedForces = std::move(o._appliedForces);
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
    _dofs.clear();
    _KPs.clear();
    _KDs.clear();
    _effortLimits.clear();
    _appliedForces.clear();
}

void Articulation::setKPs(const std::vector<float>& kps) {
    if (static_cast<int>(kps.size()) != numDofs())
        throw std::runtime_error("setKPs: size must match numDofs()");
    _KPs = kps;
}

void Articulation::setKDs(const std::vector<float>& kds) {
    if (static_cast<int>(kds.size()) != numDofs())
        throw std::runtime_error("setKDs: size must match numDofs()");
    _KDs = kds;
}

void Articulation::setEffortLimits(const std::vector<float>& effortLimits) {
    if (static_cast<int>(effortLimits.size()) != numDofs())
        throw std::runtime_error("setEffortLimits: size must match numDofs()");
    _effortLimits = effortLimits;
}

std::vector<float> Articulation::getRootPositionFlat() const {
    if (!_artic)
        return {};

    const PxVec3 p = _artic->getRootGlobalPose().p;
    return {p.x, p.y, p.z};
}

std::vector<float> Articulation::getRootRotationFlat() const {
    if (!_artic)
        return {};

    const PxQuat q = _artic->getRootGlobalPose().q;
    return {q.x, q.y, q.z, q.w};
}

std::vector<float> Articulation::getRootLinearVelocityFlat() const {
    if (!_artic)
        return {};

    const PxVec3 v = _artic->getRootLinearVelocity();
    return {v.x, v.y, v.z};
}

std::vector<float> Articulation::getRootAngularVelocityFlat() const {
    if (!_artic)
        return {};

    const PxVec3 v = _artic->getRootAngularVelocity();
    return {v.x, v.y, v.z};
}

std::vector<float> Articulation::getLinkPositionsFlat() const {
    std::vector<float> out;
    out.reserve(_links.size() * 3);
    for (const auto* link : _links) {
        if (!link) {
            out.insert(out.end(), {0.f, 0.f, 0.f});
            continue;
        }
        const PxVec3 p = link->getGlobalPose().p;
        out.insert(out.end(), {p.x, p.y, p.z});
    }
    return out;
}

std::vector<float> Articulation::getLinkRotationsFlat() const {
    std::vector<float> out;
    out.reserve(_links.size() * 4);
    for (const auto* link : _links) {
        if (!link) {
            out.insert(out.end(), {0.f, 0.f, 0.f, 1.f});
            continue;
        }
        const PxQuat q = link->getGlobalPose().q;
        out.insert(out.end(), {q.x, q.y, q.z, q.w});
    }
    return out;
}

std::vector<float> Articulation::getLinkLinearVelocitiesFlat() const {
    std::vector<float> out;
    out.reserve(_links.size() * 3);
    for (const auto* link : _links) {
        if (!link) {
            out.insert(out.end(), {0.f, 0.f, 0.f});
            continue;
        }
        const PxVec3 v = link->getLinearVelocity();
        out.insert(out.end(), {v.x, v.y, v.z});
    }
    return out;
}

std::vector<float> Articulation::getLinkAngularVelocitiesFlat() const {
    std::vector<float> out;
    out.reserve(_links.size() * 3);
    for (const auto* link : _links) {
        if (!link) {
            out.insert(out.end(), {0.f, 0.f, 0.f});
            continue;
        }
        const PxVec3 v = link->getAngularVelocity();
        out.insert(out.end(), {v.x, v.y, v.z});
    }
    return out;
}

std::vector<float> Articulation::getDofPositions() const {
    int n = numDofs();
    std::vector<float> out;
    out.reserve(n);
    if (!_artic || n == 0)
        return out;
    for (const auto& dof : _dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        out.push_back(joint ? joint->getJointPosition(dof.axis) : 0.f);
    }
    return out;
}

std::vector<float> Articulation::getDofVelocities() const {
    int n = numDofs();
    std::vector<float> out;
    out.reserve(n);
    if (!_artic || n == 0)
        return out;
    for (const auto& dof : _dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        out.push_back(joint ? joint->getJointVelocity(dof.axis) : 0.f);
    }
    return out;
}

std::vector<float> Articulation::getDofForces() const { return _appliedForces; }

std::vector<std::string> Articulation::getDofNames() const {
    std::vector<std::string> out;
    out.reserve(_dofs.size());
    for (const auto& dof : _dofs)
        out.push_back(dof.name);
    return out;
}

std::vector<std::array<float, 2>> Articulation::getDofLimits() const {
    std::vector<std::array<float, 2>> out;
    out.reserve(_dofs.size());
    for (const auto& dof : _dofs)
        out.push_back({dof.loLimit, dof.hiLimit});
    return out;
}

std::vector<float> Articulation::getDofEffortLimits() const {
    std::vector<float> out;
    out.reserve(_dofs.size());
    for (const auto& dof : _dofs)
        out.push_back(dof.effortLimit);
    return out;
}

std::vector<float> Articulation::getLinkMasses() const {
    std::vector<float> out;
    out.reserve(_links.size());
    for (const auto* link : _links)
        out.push_back(link ? link->getMass() : 0.f);
    return out;
}

float Articulation::calcMass() const {
    float total = 0.f;
    for (const auto* link : _links) {
        if (link)
            total += link->getMass();
    }
    return total;
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
                                    g0.translation.z()),
                             toPxQuat(g0.rotation)));
    {
        auto it = colGeoms.find(0);
        if (it != colGeoms.end() && !it->second.empty())
            attachCollisionShapes(artic._links[0], physics.getPhysics(),
                                  physics.getMaterial(), it->second,
                                  cfg.contactOffset, cfg.restOffset);
        else {
            PxShape* shape = PxRigidActorExt::createExclusiveShape(
                *artic._links[0],
                PxBoxGeometry(cfg.rootBoxHalf.x, cfg.rootBoxHalf.y,
                              cfg.rootBoxHalf.z),
                *physics.getMaterial());
            applyContactOffsets(shape, cfg.contactOffset, cfg.restOffset);
        }
    }
    setCollisionFilterData(artic._links[0], cfg.collisionGroup);
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
                                      globals[i].translation.z()),
                               toPxQuat(globals[i].rotation));
        PxTransform parentWorld(PxVec3(globals[pi].translation.x(),
                                       globals[pi].translation.y(),
                                       globals[pi].translation.z()),
                                toPxQuat(globals[pi].rotation));

        artic._links[i] =
            artic._artic->createLink(artic._links[pi], childWorld);
        {
            auto it = colGeoms.find(i);
            if (it != colGeoms.end() && !it->second.empty())
                attachCollisionShapes(artic._links[i], physics.getPhysics(),
                                      physics.getMaterial(), it->second,
                                      cfg.contactOffset, cfg.restOffset);
            else {
                PxShape* shape = PxRigidActorExt::createExclusiveShape(
                    *artic._links[i],
                    PxBoxGeometry(cfg.linkBoxHalf.x, cfg.linkBoxHalf.y,
                                  cfg.linkBoxHalf.z),
                    *physics.getMaterial());
                applyContactOffsets(shape, cfg.contactOffset, cfg.restOffset);
            }
        }
        setCollisionFilterData(artic._links[i], cfg.collisionGroup);
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
            artic._dofs.push_back({i, jd.name, PxArticulationAxis::eTWIST,
                                   jd.loLimit, jd.hiLimit, jd.kp, jd.kd,
                                   jd.effortLimit});
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
                    axis = PxArticulationAxis::eSWING1;
                else
                    axis = PxArticulationAxis::eSWING2;
                joint->setMotion(axis, PxArticulationMotion::eLIMITED);
                joint->setLimit(axis, jd.loLimit, jd.hiLimit);
                artic._dofs.push_back({i, jd.name, axis, jd.loLimit, jd.hiLimit,
                                       jd.kp, jd.kd, jd.effortLimit});
            }
        }
    }

    artic._KPs.reserve(artic._dofs.size());
    artic._KDs.reserve(artic._dofs.size());
    artic._effortLimits.reserve(artic._dofs.size());
    for (const auto& dof : artic._dofs) {
        artic._KPs.push_back(dof.kp);
        artic._KDs.push_back(dof.kd);
        artic._effortLimits.push_back(dof.effortLimit);
    }
    artic._appliedForces.assign(artic._dofs.size(), 0.f);

    physics.getScene()->addArticulation(*artic._artic);
    return artic;
}

// Drive
void Articulation::setDriveTargets(const std::vector<float>& targets, float kp,
                                   float kd) {
    if (!_artic)
        return;

    PxArticulationDrive drive{kp, kd, PX_MAX_F32,
                              PxArticulationDriveType::eFORCE};
    int dofIdx = 0;
    for (const auto& dof : _dofs) {
        if (dofIdx >= static_cast<int>(targets.size()))
            break;
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (!joint)
            continue;
        joint->setDriveParams(dof.axis, drive);
        joint->setDriveTarget(dof.axis, targets[dofIdx++]);
    }

    if (kp > 0.f && _artic->isSleeping())
        _artic->wakeUp();
}

void Articulation::setDriveTargets(const std::vector<float>& targets) {
    int n = numDofs();
    if (!_artic || n == 0)
        return;
    if (static_cast<int>(targets.size()) != n)
        throw std::runtime_error(
            "setDriveTargets: targets must have numDofs() entries");
    if (static_cast<int>(_KPs.size()) != n ||
        static_cast<int>(_KDs.size()) != n ||
        static_cast<int>(_effortLimits.size()) != n)
        throw std::runtime_error(
            "setDriveTargets: call setKPs/setKDs/setEffortLimits before using "
            "stored drive params");

    bool anyKp = false;
    int dofIdx = 0;
    for (const auto& dof : _dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (!joint) {
            dofIdx++;
            continue;
        }
        PxArticulationDrive drive{_KPs[dofIdx], _KDs[dofIdx],
                                  _effortLimits[dofIdx],
                                  PxArticulationDriveType::eFORCE};
        joint->setDriveParams(dof.axis, drive);
        joint->setDriveTarget(dof.axis, targets[dofIdx]);
        if (_KPs[dofIdx] > 0.f)
            anyKp = true;
        dofIdx++;
    }

    if (anyKp && _artic->isSleeping())
        _artic->wakeUp();
}

void Articulation::setDriveVelocityTargets(const std::vector<float>& targets) {
    int n = numDofs();
    if (!_artic || n == 0)
        return;
    if (static_cast<int>(targets.size()) != n)
        throw std::runtime_error(
            "setDriveVelocityTargets: targets must have numDofs() entries");
    if (static_cast<int>(_KDs.size()) != n ||
        static_cast<int>(_effortLimits.size()) != n)
        throw std::runtime_error(
            "setDriveVelocityTargets: call setKDs/setEffortLimits before using "
            "stored drive params");

    bool anyKd = false;
    int dofIdx = 0;
    for (const auto& dof : _dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (!joint) {
            dofIdx++;
            continue;
        }
        PxArticulationDrive drive{0.f, _KDs[dofIdx], _effortLimits[dofIdx],
                                  PxArticulationDriveType::eFORCE};
        joint->setDriveParams(dof.axis, drive);
        joint->setDriveVelocity(dof.axis, targets[dofIdx]);
        if (_KDs[dofIdx] > 0.f)
            anyKd = true;
        dofIdx++;
    }

    if (anyKd && _artic->isSleeping())
        _artic->wakeUp();
}

// Reset root
void Articulation::resetRoot(const PxTransform& pose) {
    setRootState(pose, PxVec3(0.f), PxVec3(0.f));
}

void Articulation::setRootState(const PxTransform& pose,
                                const PxVec3& linearVelocity,
                                const PxVec3& angularVelocity) {
    if (!_artic)
        return;

    _artic->setRootGlobalPose(pose);
    _artic->setRootLinearVelocity(linearVelocity);
    _artic->setRootAngularVelocity(angularVelocity);
    _artic->updateKinematic(PxArticulationKinematicFlag::ePOSITION |
                            PxArticulationKinematicFlag::eVELOCITY);
}

void Articulation::setDofState(const std::vector<float>& positions,
                               const std::vector<float>& velocities) {
    int n = numDofs();
    if (!_artic || n == 0)
        return;

    if (static_cast<int>(positions.size()) != n ||
        static_cast<int>(velocities.size()) != n) {
        throw std::runtime_error(
            "setDofState expected positions and velocities with numDofs() "
            "entries");
    }

    int dofIdx = 0;
    for (const auto& dof : _dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (!joint) {
            dofIdx++;
            continue;
        }
        joint->setJointPosition(dof.axis, positions[dofIdx]);
        joint->setJointVelocity(dof.axis, velocities[dofIdx]);
        dofIdx++;
    }

    if (_artic->isSleeping())
        _artic->wakeUp();
    _artic->updateKinematic(PxArticulationKinematicFlag::ePOSITION |
                            PxArticulationKinematicFlag::eVELOCITY);
}

void Articulation::setJointForces(const std::vector<float>& forces) {
    int n = numDofs();
    if (!_artic || n == 0)
        return;
    if (static_cast<int>(forces.size()) != n)
        throw std::runtime_error("setJointForces: size must match numDofs()");

    PxArticulationCache* cache = _artic->createCache();
    _artic->zeroCache(*cache);
    for (int i = 0; i < n; ++i)
        cache->jointForce[i] = forces[i];
    _artic->applyCache(*cache, PxArticulationCacheFlag::eFORCE);
    cache->release();

    _appliedForces = forces;
    if (_artic->isSleeping())
        _artic->wakeUp();
}

void Articulation::addLinkForce(int linkIndex, const PxVec3& force) {
    if (!_artic || linkIndex < 0 ||
        linkIndex >= static_cast<int>(_links.size()) || !_links[linkIndex])
        return;

    _links[linkIndex]->addForce(force, PxForceMode::eFORCE, true);
    if (_artic->isSleeping())
        _artic->wakeUp();
}

} // namespace KE

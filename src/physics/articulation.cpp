#include "articulation.hpp"
#include "animation/skeleton_state.hpp"
#include "collision_material_utils.hpp"
#include "physics/physx_compat.hpp"

#include <Eigen/Geometry>
#include <extensions/PxRigidBodyExt.h>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace KE {
namespace {

constexpr PxU32 kInvalidLinkIndex = std::numeric_limits<PxU32>::max();

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

struct MultiAxisFrame {
    PxQuat rotation = PxQuat(PxIdentity);
    std::vector<PxArticulationAxis::Enum> axes;
};

// PhysX spherical joints expose three orthogonal axes in one joint frame.
// MJCF instead authors each hinge axis in body-local space. Build a common
// right-handed frame so arbitrary signed/rotated orthogonal axes (for example
// nv_humanoid's diagonal shoulder axes) remain distinct PhysX DOFs.
MultiAxisFrame makeMultiAxisFrame(
    const std::vector<Character::JointDesc>& joints) {
    if (joints.size() < 2 || joints.size() > 3)
        throw std::runtime_error(
            "multi-axis articulation joints require two or three axes");

    std::vector<Eigen::Vector3f> authoredAxes;
    authoredAxes.reserve(joints.size());
    for (const auto& joint : joints) {
        if (joint.axis.squaredNorm() < 1e-12f)
            throw std::runtime_error("articulation joint axis must be non-zero");
        authoredAxes.push_back(joint.axis.normalized());
    }
    for (size_t i = 0; i < authoredAxes.size(); ++i) {
        for (size_t j = i + 1; j < authoredAxes.size(); ++j) {
            if (std::abs(authoredAxes[i].dot(authoredAxes[j])) > 0.02f)
                throw std::runtime_error(
                    "multi-axis articulation joint axes must be orthogonal");
        }
    }

    Eigen::Matrix3f basis;
    basis.col(0) = authoredAxes[0];
    MultiAxisFrame result;
    result.axes.resize(joints.size());
    result.axes[0] = PxArticulationAxis::eTWIST;

    if (joints.size() == 2) {
        basis.col(1) = authoredAxes[1];
        basis.col(2) = basis.col(0).cross(basis.col(1)).normalized();
        result.axes[1] = PxArticulationAxis::eSWING1;
    } else {
        const Eigen::Vector3f cross =
            authoredAxes[0].cross(authoredAxes[1]).normalized();
        if (cross.dot(authoredAxes[2]) >= 0.f) {
            basis.col(1) = authoredAxes[1];
            basis.col(2) = authoredAxes[2];
            result.axes[1] = PxArticulationAxis::eSWING1;
            result.axes[2] = PxArticulationAxis::eSWING2;
        } else {
            // The authored order is left-handed. Swap the last two PhysX axes
            // while preserving the public/logical MJCF DOF order.
            basis.col(1) = authoredAxes[2];
            basis.col(2) = authoredAxes[1];
            result.axes[1] = PxArticulationAxis::eSWING2;
            result.axes[2] = PxArticulationAxis::eSWING1;
        }
    }

    Eigen::Quaternionf q(basis);
    q.normalize();
    result.rotation = toPxQuat(q);
    return result;
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

Character::CollisionGeomDesc makeFallbackBoxGeom(const PxVec3& halfExtents) {
    Character::CollisionGeomDesc geom;
    geom.type = Character::CollisionGeomDesc::Type::Box;
    geom.name = "__fallback_box";
    geom.size[0] = halfExtents.x;
    geom.size[1] = halfExtents.y;
    geom.size[2] = halfExtents.z;
    geom.isFallback = true;
    return geom;
}

// Creates and attaches PhysX shapes for each MJCF collision geom on the link.
// Cylinders are approximated as capsules (PhysX has no native cylinder shape).
void attachCollisionShapes(PxArticulationLink* link, PhysicsWorld& physics,
                           const Character::CollisionGeomDesc* geoms,
                           std::size_t count, float contactOffset,
                           float restOffset,
                           const std::vector<
                               Physics::CollisionMaterialOverride>& overrides,
                           std::shared_ptr<const Animation::SkeletonTree> tree,
                           int bodyIndex) {
    using Type = Character::CollisionGeomDesc::Type;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& g = geoms[i];
        const auto material =
            resolveCollisionMaterial(g, overrides, tree, bodyIndex,
                                     static_cast<int>(i));
        PxShape* shape = nullptr;
        PxTransform localPose(PxIdentity);
        switch (g.type) {
        case Type::Capsule:
        case Type::Cylinder: {
            float radius = g.size[0];
            if (g.hasFromTo) {
                float halfH = (g.to - g.from).norm() * 0.5f;
                shape = physics.createExclusiveShape(
                    *link, PxCapsuleGeometry(radius, halfH), material);
                localPose = fromToPose(g.from, g.to);
            } else {
                shape = physics.createExclusiveShape(
                    *link, PxCapsuleGeometry(radius, g.size[1]), material);
                localPose = PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                                        mjcfShapeRot(g.quat));
            }
            break;
        }
        case Type::Sphere:
            shape = physics.createExclusiveShape(
                *link, PxSphereGeometry(g.size[0]), material);
            localPose = PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()));
            break;
        case Type::Box:
            shape = physics.createExclusiveShape(
                *link, PxBoxGeometry(g.size[0], g.size[1], g.size[2]),
                material);
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
    }
}

// Convenience overload accepting a vector of geoms.
void attachCollisionShapes(PxArticulationLink* link, PhysicsWorld& physics,
                           const std::vector<Character::CollisionGeomDesc>& geoms,
                           float contactOffset, float restOffset,
                           const std::vector<
                               Physics::CollisionMaterialOverride>& overrides,
                           std::shared_ptr<const Animation::SkeletonTree> tree,
                           int bodyIndex) {
    attachCollisionShapes(link, physics, geoms.data(), geoms.size(),
                          contactOffset, restOffset, overrides, tree, bodyIndex);
}

// Applies MJCF inertial properties (mass, COM, diag inertia) to a link.
// Falls back to uniform mass distribution if the link has no inertial entry.
void applyInertial(PxArticulationLink* link,
                   const Character::InertialDescMap& inertials, int idx,
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
    : _artic(o._artic), _aggregate(o._aggregate),
      _links(std::move(o._links)),
      _template(std::move(o._template)),
      _KPs(std::move(o._KPs)), _KDs(std::move(o._KDs)),
      _effortLimits(std::move(o._effortLimits)),
      _appliedForces(std::move(o._appliedForces)) {
    o._artic = nullptr;
    o._aggregate = nullptr;
}

Articulation& Articulation::operator=(Articulation&& o) noexcept {
    if (this != &o) {
        release();
        _artic = o._artic;
        _aggregate = o._aggregate;
        _links = std::move(o._links);
        _template = std::move(o._template);
        _KPs = std::move(o._KPs);
        _KDs = std::move(o._KDs);
        _effortLimits = std::move(o._effortLimits);
        _appliedForces = std::move(o._appliedForces);
        o._artic = nullptr;
        o._aggregate = nullptr;
    }
    return *this;
}

Articulation::~Articulation() { release(); }

void Articulation::release() {
    if (_artic) {
        _artic->release();
        _artic = nullptr;
    }
    if (_aggregate) {
        _aggregate->release();
        _aggregate = nullptr;
    }
    _links.clear();
    _template.reset();
    _KPs.clear();
    _KDs.clear();
    _effortLimits.clear();
    _appliedForces.clear();
}

void Articulation::setKPs(const std::vector<float>& kps) {
    if (static_cast<int>(kps.size()) != numDofs())
        throw std::runtime_error("setKPs: size must match numDofs()");
    _KPs = kps;
    syncDriveParams();
}

void Articulation::setKDs(const std::vector<float>& kds) {
    if (static_cast<int>(kds.size()) != numDofs())
        throw std::runtime_error("setKDs: size must match numDofs()");
    _KDs = kds;
    syncDriveParams();
}

void Articulation::setEffortLimits(const std::vector<float>& effortLimits) {
    if (static_cast<int>(effortLimits.size()) != numDofs())
        throw std::runtime_error("setEffortLimits: size must match numDofs()");
    _effortLimits = effortLimits;
    syncDriveParams();
}

void Articulation::syncDriveParams() {
    const int n = numDofs();
    if (!_artic || n == 0)
        return;
    if (static_cast<int>(_KPs.size()) != n ||
        static_cast<int>(_KDs.size()) != n ||
        static_cast<int>(_effortLimits.size()) != n)
        return;

    int dofIdx = 0;
    for (const auto& dof : _template->_dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (joint) {
            PhysXCompat::setArticulationDrive(*joint, dof.axis, _KPs[dofIdx],
                                              _KDs[dofIdx],
                                              _effortLimits[dofIdx]);
        }
        dofIdx++;
    }
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

std::vector<int> Articulation::getLinkIndices() const {
    std::vector<int> result;
    result.reserve(_links.size());
    for (const auto* link : _links) {
        const PxU32 index = link ? link->getLinkIndex() : kInvalidLinkIndex;
        if (index == kInvalidLinkIndex)
            throw std::runtime_error("articulation link index is unavailable "
                                     "before scene insertion");
        result.push_back(static_cast<int>(index));
    }
    return result;
}

std::vector<float> Articulation::getDofPositions() const {
    int n = numDofs();
    std::vector<float> out;
    out.reserve(n);
    if (!_artic || n == 0)
        return out;
    for (const auto& dof : _template->_dofs) {
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
    for (const auto& dof : _template->_dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        out.push_back(joint ? joint->getJointVelocity(dof.axis) : 0.f);
    }
    return out;
}

std::vector<float> Articulation::getDofForces() const { return _appliedForces; }

std::vector<std::string> Articulation::getDofNames() const {
    std::vector<std::string> out;
    if (!_template)
        return out;
    out.reserve(_template->_dofs.size());
    for (const auto& dof : _template->_dofs)
        out.push_back(dof.name);
    return out;
}

std::vector<int> Articulation::getDofGpuIndices() const {
    // PhysX Direct GPU API stores articulation DOFs in low-level link index
    // order, then in PxArticulationAxis enum order within each inbound joint.
    // KangEngine exposes DOFs in logical/skeleton order, so return a scatter
    // map: logical DOF index -> PhysX GPU buffer column.
    if (!_template)
        return {};
    std::vector<int> order(_template->_dofs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [this](int a, int b) {
        const auto& da = _template->_dofs[static_cast<size_t>(a)];
        const auto& db = _template->_dofs[static_cast<size_t>(b)];
        const PxU32 la =
            _links[static_cast<size_t>(da.linkIndex)]->getLinkIndex();
        const PxU32 lb =
            _links[static_cast<size_t>(db.linkIndex)]->getLinkIndex();
        if (la != lb)
            return la < lb;
        return static_cast<int>(da.axis) < static_cast<int>(db.axis);
    });

    std::vector<int> result(_template->_dofs.size(), -1);
    for (size_t gpuIndex = 0; gpuIndex < order.size(); ++gpuIndex)
        result[static_cast<size_t>(order[gpuIndex])] =
            static_cast<int>(gpuIndex);
    return result;
}

std::vector<std::array<float, 2>> Articulation::getDofLimits() const {
    std::vector<std::array<float, 2>> out;
    if (!_template)
        return out;
    out.reserve(_template->_dofs.size());
    for (const auto& dof : _template->_dofs)
        out.push_back({dof.loLimit, dof.hiLimit});
    return out;
}

std::vector<float> Articulation::getDofEffortLimits() const {
    std::vector<float> out;
    if (!_template)
        return out;
    out.reserve(_template->_dofs.size());
    for (const auto& dof : _template->_dofs)
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

std::shared_ptr<ArticulationTemplate> ArticulationTemplate::create(
    std::shared_ptr<const Animation::SkeletonTree> tree,
    const Character::CollisionGeomDescMap& colGeoms,
    const Character::JointDescMap& joints,
    const Character::InertialDescMap& inertials,
    const ArticulationConfig& cfg) {
    if (!tree || tree->numJoints() == 0)
        throw std::runtime_error(
            "ArticulationTemplate requires a non-empty SkeletonTree");

    auto result = std::make_shared<ArticulationTemplate>();
    result->_tree = std::move(tree);
    result->_joints = joints;
    result->_colGeoms = colGeoms;
    result->_inertials = inertials;

    auto state = Animation::SkeletonState::zeroPose(result->_tree);
    const auto globals = state.computeGlobalTransforms();
    const int n = result->_tree->numJoints();
    result->_restTransforms.reserve(static_cast<size_t>(n));
    result->_jointFrames.assign(static_cast<size_t>(n), PxQuat(PxIdentity));
    result->_bodyNames.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& transform = globals[static_cast<size_t>(i)];
        result->_restTransforms.emplace_back(
            PxVec3(transform.translation.x(), transform.translation.y(),
                   transform.translation.z()),
            toPxQuat(transform.rotation));
        result->_bodyNames.push_back(result->_tree->nodeName(i));

        auto jit = result->_joints.find(i);
        if (jit == result->_joints.end())
            continue;
        if (jit->second.size() == 1) {
            const auto& jd = jit->second[0];
            result->_jointFrames[static_cast<size_t>(i)] =
                axisAlignQuat(jd.axis);
            result->_dofs.push_back(
                {i, jd.name, PxArticulationAxis::eTWIST, jd.loLimit,
                 jd.hiLimit, jd.kp, jd.kd, jd.effortLimit});
            continue;
        }

        const MultiAxisFrame frame = makeMultiAxisFrame(jit->second);
        result->_jointFrames[static_cast<size_t>(i)] = frame.rotation;
        for (size_t jointIndex = 0; jointIndex < jit->second.size();
             ++jointIndex) {
            const auto& jd = jit->second[jointIndex];
            const auto axis = frame.axes[jointIndex];
            result->_dofs.push_back({i, jd.name, axis, jd.loLimit, jd.hiLimit,
                                     jd.kp, jd.kd, jd.effortLimit});
        }
    }
    for (int i = 0; i < n; ++i) {
        auto it = result->_colGeoms.find(i);
        if (it != result->_colGeoms.end() && it->second.empty())
            it->second.push_back(makeFallbackBoxGeom(
                i == 0 ? cfg.rootBoxHalf : cfg.linkBoxHalf));
    }
    return result;
}

Articulation Articulation::build(
    PhysicsWorld& physics, std::shared_ptr<const Animation::SkeletonTree> tree,
    const Character::CollisionGeomDescMap& colGeoms,
    const Character::JointDescMap& joints,
    const Character::InertialDescMap& inertials,
    const ArticulationConfig& cfg) {
    return build(physics,
                 ArticulationTemplate::create(std::move(tree), colGeoms, joints,
                                              inertials, cfg),
                 cfg);
}

Articulation Articulation::build(
    PhysicsWorld& physics,
    std::shared_ptr<ArticulationTemplate> articulationTemplate,
    const ArticulationConfig& cfg) {
    if (!articulationTemplate)
        throw std::runtime_error("Articulation::build requires a template");

    Articulation artic;
    artic._template = std::move(articulationTemplate);
    const auto& tree = artic._template->_tree;
    const auto& joints = artic._template->_joints;
    const auto& colGeoms = artic._template->_colGeoms;
    const auto& inertials = artic._template->_inertials;
    const auto& globals = artic._template->_restTransforms;
    const int n = artic._template->numLinks();

    artic._artic = physics.getPhysics()->createArticulationReducedCoordinate();
    artic._artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE,
                                      cfg.fixBase);
    artic._artic->setArticulationFlag(
        PxArticulationFlag::eDISABLE_SELF_COLLISION, cfg.disableSelfCollision);
    artic._artic->setSolverIterationCounts(
        static_cast<PxU32>(cfg.solverPositionIterations),
        static_cast<PxU32>(cfg.solverVelocityIterations));
    artic._links.resize(n, nullptr);

    artic._links[0] = artic._artic->createLink(nullptr, globals[0]);
    auto rootGeoms = colGeoms.find(0);
    if (rootGeoms != colGeoms.end())
        attachCollisionShapes(artic._links[0], physics, rootGeoms->second,
                              cfg.contactOffset, cfg.restOffset,
                              cfg.materialOverrides, tree, 0);
    setCollisionFilterData(artic._links[0], cfg.collisionGroup);
    applyInertial(artic._links[0], inertials, 0, cfg.defaultRootMass);
    artic._links[0]->setLinearDamping(cfg.rootLinearDamping);
    artic._links[0]->setAngularDamping(cfg.rootAngularDamping);
    if (cfg.enableCCD)
        artic._links[0]->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);

    for (int i = 1; i < n; ++i) {
        const int pi = tree->parentIndex(i);
        const PxTransform& childWorld = globals[static_cast<size_t>(i)];
        const PxTransform& parentWorld = globals[static_cast<size_t>(pi)];
        artic._links[i] =
            artic._artic->createLink(artic._links[pi], childWorld);

        auto geomIt = colGeoms.find(i);
        if (geomIt != colGeoms.end())
            attachCollisionShapes(artic._links[i], physics, geomIt->second,
                                  cfg.contactOffset, cfg.restOffset,
                                  cfg.materialOverrides, tree, i);
        setCollisionFilterData(artic._links[i], cfg.collisionGroup);
        applyInertial(artic._links[i], inertials, i, cfg.defaultLinkMass);
        artic._links[i]->setLinearDamping(cfg.linkLinearDamping);
        artic._links[i]->setAngularDamping(cfg.linkAngularDamping);
        if (cfg.maxAngularVelocity > 0.f)
            artic._links[i]->setMaxAngularVelocity(cfg.maxAngularVelocity);

        auto* joint = static_cast<PxArticulationJointReducedCoordinate*>(
            artic._links[i]->getInboundJoint());
        auto jit = joints.find(i);
        const int ndof =
            jit != joints.end() ? static_cast<int>(jit->second.size()) : 0;
        if (ndof == 0) {
            joint->setJointType(PxArticulationJointType::eFIX);
            joint->setParentPose(parentWorld.getInverse() * childWorld);
            joint->setChildPose(PxTransform(PxIdentity));
        } else if (ndof == 1) {
            const auto& jd = jit->second[0];
            joint->setJointType(PxArticulationJointType::eREVOLUTE);
            const PxTransform childPose(
                PxVec3(0.f),
                artic._template->_jointFrames[static_cast<size_t>(i)]);
            joint->setParentPose(parentWorld.getInverse() * childWorld *
                                 childPose);
            joint->setChildPose(childPose);
            joint->setMotion(PxArticulationAxis::eTWIST,
                             PxArticulationMotion::eLIMITED);
            PhysXCompat::setArticulationLimit(
                *joint, PxArticulationAxis::eTWIST, jd.loLimit, jd.hiLimit);
        } else {
            joint->setJointType(PxArticulationJointType::eSPHERICAL);
            const PxTransform childPose(
                PxVec3(0.f),
                artic._template->_jointFrames[static_cast<size_t>(i)]);
            joint->setParentPose(parentWorld.getInverse() * childWorld *
                                 childPose);
            joint->setChildPose(childPose);
            for (const auto& dof : artic._template->_dofs) {
                if (dof.linkIndex != i)
                    continue;
                joint->setMotion(dof.axis, PxArticulationMotion::eLIMITED);
                PhysXCompat::setArticulationLimit(
                    *joint, dof.axis, dof.loLimit, dof.hiLimit);
            }
        }
    }

    for (const auto& dof : artic._template->_dofs) {
        artic._KPs.push_back(dof.kp);
        artic._KDs.push_back(dof.kd);
        artic._effortLimits.push_back(dof.effortLimit);
    }
    artic._appliedForces.assign(artic._template->_dofs.size(), 0.f);
    artic.syncDriveParams();
    if (cfg.useAggregate) {
        PxU32 shapeCount = 0;
        for (auto* link : artic._links)
            shapeCount += link->getNbShapes();
        const auto filterHint = PxGetAggregateFilterHint(
            PxAggregateType::eGENERIC, !cfg.disableSelfCollision);
        artic._aggregate =
            physics.getPhysics()->createAggregate(n, shapeCount, filterHint);
        if (!artic._aggregate ||
            !artic._aggregate->addArticulation(*artic._artic)) {
            throw std::runtime_error(
                "Failed to add articulation to PhysX aggregate");
        }
        physics.getScene()->addAggregate(*artic._aggregate);
    } else {
        physics.getScene()->addArticulation(*artic._artic);
    }
    const PxU32 physxDofCount = artic._artic->getDofs();
    const PxU32 logicalDofCount =
        static_cast<PxU32>(artic._template->_dofs.size());
    if (physxDofCount != logicalDofCount) {
        throw std::runtime_error(
            "Articulation DOF mapping produced " +
            std::to_string(physxDofCount) + " PhysX DOFs for " +
            std::to_string(logicalDofCount) +
            " authored DOFs; check multi-axis joint definitions");
    }
    return artic;
}

// Drive
void Articulation::setDriveTargets(const std::vector<float>& targets, float kp,
                                   float kd) {
    if (!_artic)
        return;

    int dofIdx = 0;
    for (const auto& dof : _template->_dofs) {
        if (dofIdx >= static_cast<int>(targets.size()))
            break;
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (!joint)
            continue;
        PhysXCompat::setArticulationDrive(*joint, dof.axis, kp, kd, PX_MAX_F32);
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
    for (const auto& dof : _template->_dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (!joint) {
            dofIdx++;
            continue;
        }
        PhysXCompat::setArticulationDrive(*joint, dof.axis, _KPs[dofIdx],
                                          _KDs[dofIdx], _effortLimits[dofIdx]);
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
    for (const auto& dof : _template->_dofs) {
        auto* joint = inboundJoint(_links, dof.linkIndex);
        if (!joint) {
            dofIdx++;
            continue;
        }
        PhysXCompat::setArticulationDrive(*joint, dof.axis, 0.f, _KDs[dofIdx],
                                          _effortLimits[dofIdx]);
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
    for (const auto& dof : _template->_dofs) {
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

int Articulation::setCollisionMaterial(
    PhysicsWorld& physics, const Physics::PhysicsMaterialDesc& material) {
    if (!physics.getPhysics() || !_artic)
        return 0;

    int updated = 0;
    for (auto* link : _links) {
        if (!link)
            continue;
        const PxU32 shapeCount = link->getNbShapes();
        if (shapeCount == 0)
            continue;

        std::vector<PxShape*> shapes(shapeCount);
        link->getShapes(shapes.data(), shapeCount);
        for (auto* shape : shapes) {
            if (!shape)
                continue;
            PxMaterial* shapeMat = physics.getMaterialForDesc(material);
            shape->setMaterials(&shapeMat, 1);
            ++updated;
        }
    }
    return updated;
}

int Articulation::setCollisionMaterialOverrides(
    PhysicsWorld& physics,
    const std::vector<Physics::CollisionMaterialOverride>& overrides) {
    if (!physics.getPhysics() || !_artic)
        return 0;

    int updated = 0;
    for (const auto& [bodyIndex, geoms] : _template->_colGeoms) {
        if (bodyIndex < 0 || bodyIndex >= static_cast<int>(_links.size()))
            continue;
        auto* link = _links[static_cast<size_t>(bodyIndex)];
        if (!link)
            continue;

        const PxU32 shapeCount = link->getNbShapes();
        if (shapeCount == 0 || geoms.empty())
            continue;

        std::vector<PxShape*> shapes(shapeCount);
        link->getShapes(shapes.data(), shapeCount);
        const std::size_t count =
            std::min<std::size_t>(geoms.size(), shapes.size());
        for (std::size_t i = 0; i < count; ++i) {
            auto* shape = shapes[i];
            if (!shape)
                continue;
            const auto material = resolveCollisionMaterial(
                geoms[i], overrides, _template->_bodyNames, bodyIndex,
                static_cast<int>(i));
            PxMaterial* shapeMat = physics.getMaterialForDesc(material);
            shape->setMaterials(&shapeMat, 1);
            ++updated;
        }
    }
    return updated;
}

void Articulation::addLinkForce(int linkIndex, const PxVec3& force) {
    if (!_artic || linkIndex < 0 ||
        linkIndex >= static_cast<int>(_links.size()) || !_links[linkIndex])
        return;

    _links[linkIndex]->addForce(force, PxForceMode::eFORCE, true);
    if (_artic->isSleeping())
        _artic->wakeUp();
}

void Articulation::addLinkForceAtPosition(int linkIndex, const PxVec3& force,
                                          const PxVec3& position) {
    if (!_artic || linkIndex < 0 ||
        linkIndex >= static_cast<int>(_links.size()) || !_links[linkIndex])
        return;

    PxRigidBodyExt::addForceAtPos(*_links[linkIndex], force, position,
                                  PxForceMode::eFORCE, true);
    if (_artic->isSleeping())
        _artic->wakeUp();
}

} // namespace KE

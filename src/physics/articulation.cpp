#include "articulation.hpp"

#include "scene/native/prim.hpp"
#include "scene/scene_backend.hpp"
#include <Eigen/Geometry>
#include <glm/gtc/quaternion.hpp>
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
                           const Animation::MJCFCollisionGeom* geoms,
                           std::size_t count) {
    using Type = Animation::MJCFCollisionGeom::Type;
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
void attachCollisionShapes(
    PxArticulationLink* link, PxMaterial* mat,
    const std::vector<Animation::MJCFCollisionGeom>& geoms) {
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
      _joints(std::move(o._joints)), _colGeoms(std::move(o._colGeoms)),
      _colVisuals(std::move(o._colVisuals)) {
    o._artic = nullptr;
}

Articulation& Articulation::operator=(Articulation&& o) noexcept {
    if (this != &o) {
        if (_artic)
            _artic->release();
        _artic = o._artic;
        _links = std::move(o._links);
        _joints = std::move(o._joints);
        _colGeoms = std::move(o._colGeoms);
        _colVisuals = std::move(o._colVisuals);
        o._artic = nullptr;
    }
    return *this;
}

Articulation::~Articulation() {
    if (_artic)
        _artic->release();
}

Articulation Articulation::build(PhysicsWorld& physics,
                                 Animation::SkelMesh& skelMesh,
                                 const std::string& mjcfPath,
                                 const ArticulationConfig& cfg) {
    const auto& skelTree = skelMesh.skeleton();
    const auto joints = Animation::parseMJCFJoints(mjcfPath, skelTree);
    const auto colGeoms = Animation::parseMJCFCollision(mjcfPath, skelTree);
    const auto inertials = Animation::parseMJCFInertial(mjcfPath, skelTree);
    const auto globals = skelMesh.state().computeGlobalTransforms();

    Articulation artic;
    artic._joints = joints;
    artic._colGeoms = colGeoms;
    int n = skelTree.numJoints();

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
        int pi = skelTree.parentIndex(i);
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
        joint->setJointType(PxArticulationJointType::eREVOLUTE);

        const auto& jd = joints[i];
        PxQuat aq = axisAlignQuat(jd.axis);
        PxTransform childPose(PxVec3(0.f), aq);
        PxTransform parentPose =
            parentWorld.getInverse() * childWorld * childPose;
        joint->setParentPose(parentPose);
        joint->setChildPose(childPose);
        joint->setMotion(PxArticulationAxis::eTWIST,
                         PxArticulationMotion::eLIMITED);
        joint->setLimit(PxArticulationAxis::eTWIST, jd.loLimit, jd.hiLimit);
    }

    for (int i = 0; i < n; i++) {
        PhysicsWorld::RigidVisual rv;
        rv.link = artic._links[i];
        rv.prim = skelMesh.bodyPrim(i);
        physics.registerRigidVisual(rv);
    }

    physics.getScene()->addArticulation(*artic._artic);
    return artic;
}

// Drive
void Articulation::setDriveTargets(const std::vector<float>& targets, float kp,
                                   float kd) {
    PxArticulationDrive drive;
    drive.stiffness = kp;
    drive.damping = kd;
    drive.maxForce = PX_MAX_F32;

    int n = static_cast<int>(_links.size());
    for (int i = 1; i < n; i++) {
        auto* joint = static_cast<PxArticulationJointReducedCoordinate*>(
            _links[i]->getInboundJoint());
        joint->setDriveParams(PxArticulationAxis::eTWIST, drive);
        joint->setDriveTarget(PxArticulationAxis::eTWIST, targets[i - 1]);
    }
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

// Collision visualization

std::vector<Scene::Prim*>
Articulation::buildCollisionVisuals(Scene::SceneBackend* scene,
                                    const std::string& basePath) {
    std::vector<Scene::Prim*> result;
    _colVisuals.clear();

    for (auto& [bodyIdx, geoms] : _colGeoms) {
        if (bodyIdx >= static_cast<int>(_links.size()))
            continue;
        PxArticulationLink* lnk = _links[bodyIdx];

        for (int gi = 0; gi < static_cast<int>(geoms.size()); gi++) {
            const auto& geom = geoms[gi];
            std::string path = basePath + "/b" + std::to_string(bodyIdx) +
                               "_g" + std::to_string(gi);
            auto* prim = scene->definePrim(path, Scene::PrimType::Mesh);

            Scene::MeshData meshData;
            glm::vec3 localPos{0.f};
            glm::quat localQuat{1.f, 0.f, 0.f, 0.f};

            if (geom.hasFromTo) {
                Eigen::Vector3f center = (geom.from + geom.to) * 0.5f;
                Eigen::Vector3f axis = (geom.to - geom.from).normalized();
                float halfLen = (geom.to - geom.from).norm() * 0.5f;
                Eigen::Quaternionf eq = Eigen::Quaternionf::FromTwoVectors(
                    Eigen::Vector3f::UnitZ(), axis);
                localPos = glm::vec3(center.x(), center.y(), center.z());
                localQuat = glm::quat(eq.w(), eq.x(), eq.y(), eq.z());

                float r = geom.size[0];
                if (geom.type == Animation::MJCFCollisionGeom::Type::Capsule)
                    meshData = Scene::Prim::createCapsuleData(r, halfLen * 2.f,
                                                              UpAxis::Z, 12);
                else
                    meshData = Scene::Prim::createCylinderData(r, halfLen * 2.f,
                                                               UpAxis::Z, 12);
            } else {
                localPos = glm::vec3(geom.pos.x(), geom.pos.y(), geom.pos.z());
                localQuat = glm::quat(geom.quat.w(), geom.quat.x(),
                                      geom.quat.y(), geom.quat.z());

                switch (geom.type) {
                case Animation::MJCFCollisionGeom::Type::Sphere:
                    meshData =
                        Scene::Prim::createSphereData(geom.size[0], 12, 8);
                    break;
                case Animation::MJCFCollisionGeom::Type::Capsule:
                    meshData = Scene::Prim::createCapsuleData(
                        geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                    break;
                case Animation::MJCFCollisionGeom::Type::Cylinder:
                    meshData = Scene::Prim::createCylinderData(
                        geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                    break;
                case Animation::MJCFCollisionGeom::Type::Box:
                    meshData = Scene::Prim::createRectangleData(
                        geom.size[0] * 2.f, geom.size[1] * 2.f,
                        geom.size[2] * 2.f);
                    break;
                }
            }

            prim->setMeshData(
                std::make_shared<Scene::MeshData>(std::move(meshData)));
            prim->setDisplayColorAlpha(glm::vec4(1.f, 0.5f, 0.f, 0.8f));
            prim->addTranslateOp(localPos);
            prim->addRotateQuaternionOp(localQuat);
            prim->setVisible(false);

            _colVisuals.push_back({lnk, prim, localPos, localQuat});
            result.push_back(prim);
        }
    }
    return result;
}

void Articulation::syncCollisionVisuals() {
    for (auto& cv : _colVisuals) {
        PxTransform pose = cv.link->getGlobalPose();
        glm::vec3 linkPos = pxToGlm(pose.p);
        glm::quat linkRot = pxToGlm(pose.q);
        cv.prim->setAttribute("xformOp:translate",
                              linkPos + linkRot * cv.localPos);
        cv.prim->setAttribute("xformOp:rotateQuaternion",
                              linkRot * cv.localQuat);
    }
}

void Articulation::setCollisionVisible(bool visible) {
    for (auto& cv : _colVisuals)
        cv.prim->setVisible(visible);
}

} // namespace KE

#include "physics.hpp"
#include "animation/character_description.hpp"
#include "articulation.hpp"
#include "foundation/Px.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace KE {

namespace {

PxTransform rigidFromToPose(const Eigen::Vector3f& from,
                            const Eigen::Vector3f& to) {
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

PxQuat rigidMjcfShapeRot(const Eigen::Quaternionf& mjcfQuat) {
    Eigen::Vector3f axis = mjcfQuat * Eigen::Vector3f::UnitZ();
    Eigen::Quaternionf q =
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), axis);
    return PxQuat(q.x(), q.y(), q.z(), q.w());
}

void applyRigidContactOffsets(PxShape* shape, float contactOffset,
                              float restOffset) {
    if (!shape)
        return;
    shape->setRestOffset(restOffset);
    shape->setContactOffset(std::max(contactOffset, restOffset + 1e-4f));
}

void setRigidCollisionFilterData(PxRigidActor* actor, PxU32 collisionGroup) {
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

} // namespace

static PxFilterFlags
contactReportFilterShader(PxFilterObjectAttributes attributes0,
                          PxFilterData filterData0,
                          PxFilterObjectAttributes attributes1,
                          PxFilterData filterData1, PxPairFlags& pairFlags,
                          const void* constantBlock, PxU32 constantBlockSize) {
    PxFilterFlags flags = PxDefaultSimulationFilterShader(
        attributes0, filterData0, attributes1, filterData1, pairFlags,
        constantBlock, constantBlockSize);

    if (filterData0.word3 != 0 && filterData1.word3 != 0 &&
        filterData0.word2 != filterData1.word2) {
        return PxFilterFlag::eSUPPRESS;
    }

    pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                 PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                 PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return flags;
}

class PhysicsWorld::ContactReportCallback : public PxSimulationEventCallback {
  public:
    explicit ContactReportCallback(std::vector<ContactPoint>& contacts)
        : _contacts(contacts) {}

    void onConstraintBreak(PxConstraintInfo* constraints,
                           PxU32 count) override {
        PX_UNUSED(constraints);
        PX_UNUSED(count);
    }
    void onWake(PxActor** actors, PxU32 count) override {
        PX_UNUSED(actors);
        PX_UNUSED(count);
    }
    void onSleep(PxActor** actors, PxU32 count) override {
        PX_UNUSED(actors);
        PX_UNUSED(count);
    }
    void onTrigger(PxTriggerPair* pairs, PxU32 count) override {
        PX_UNUSED(pairs);
        PX_UNUSED(count);
    }
    void onAdvance(const PxRigidBody* const* bodyBuffer,
                   const PxTransform* poseBuffer, const PxU32 count) override {
        PX_UNUSED(bodyBuffer);
        PX_UNUSED(poseBuffer);
        PX_UNUSED(count);
    }

    void onContact(const PxContactPairHeader& pairHeader,
                   const PxContactPair* pairs, PxU32 nbPairs) override {
        std::vector<PxContactPairPoint> contactPoints;

        for (PxU32 i = 0; i < nbPairs; ++i) {
            const PxU32 contactCount = pairs[i].contactCount;
            if (contactCount == 0)
                continue;

            contactPoints.resize(contactCount);
            pairs[i].extractContacts(contactPoints.data(), contactCount);

            for (PxU32 j = 0; j < contactCount; ++j) {
                const PxContactPairPoint& p = contactPoints[j];
                ContactPoint contact;
                contact.position = pxToGlm(p.position);
                contact.normal = pxToGlm(p.normal);
                contact.impulse = pxToGlm(p.impulse);
                contact.separation = p.separation;
                contact.actor0 = pairHeader.actors[0];
                contact.actor1 = pairHeader.actors[1];
                _contacts.push_back(contact);
            }
        }
    }

  private:
    std::vector<ContactPoint>& _contacts;
};

PhysicsWorld::PhysicsWorld(PhysicsConfig config) {
    _dt = config.dt;
    _upAxis = config.upAxis;
    _gravity = PxVec3(config.gravity[0], config.gravity[1], config.gravity[2]);
    _friction =
        PxVec3(config.friction[0], config.friction[1], config.friction[2]);

    _foundation =
        PxCreateFoundation(PX_PHYSICS_VERSION, _allocator, _errorCallback);
    _physics = PxCreatePhysics(PX_PHYSICS_VERSION, *_foundation,
                               PxTolerancesScale(), true);
    PxInitExtensions(*_physics, nullptr);

    PxSceneDesc sceneDesc(_physics->getTolerancesScale());
    sceneDesc.gravity = _gravity;
    _dispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.cpuDispatcher = _dispatcher;
    sceneDesc.filterShader = config.filterShader;
    if (config.enableContactReports) {
        _contactCallback = std::make_unique<ContactReportCallback>(_contacts);
        sceneDesc.simulationEventCallback = _contactCallback.get();
        if (config.filterShader == PxDefaultSimulationFilterShader)
            sceneDesc.filterShader = contactReportFilterShader;
    }
    sceneDesc.solverType = config.solverType;
    _scene = _physics->createScene(sceneDesc);

    _material = _physics->createMaterial(
        _friction[0], _friction[1],
        _friction[2]); // staticFriction, dynamicFriction, restitution
    _material->setFrictionCombineMode(PxCombineMode::eMIN);
    fmt::print("PhysX is initialized.\n");
}

PhysicsWorld::~PhysicsWorld() {
    if (_scene)
        _scene->release();
    if (_dispatcher)
        _dispatcher->release();
    if (_material)
        _material->release();
    _contactCallback.reset();
    PxCloseExtensions();
    if (_physics)
        _physics->release();
    if (_foundation)
        _foundation->release();
};

void PhysicsWorld::step() {
    clearContacts();
    _scene->simulate(_dt); // _dt is already deltaTime (1/60)
    _scene->fetchResults(true);
}

void PhysicsWorld::fecthData() {
    // TODO: complete me
    PxU32 nbActors = _scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
}

void PhysicsWorld::addDefaultGround() {
    PxVec3 normal = (_upAxis == UpAxis::Z) ? PxVec3(0, 0, 1) : PxVec3(0, 1, 0);
    PxRigidStatic* groundPlane =
        PxCreatePlane(*_physics, PxPlane(normal, 0), *_material);
    _scene->addActor(*groundPlane);
    registerGroundActor(groundPlane);
}

void PhysicsWorld::registerGroundActor(const PxActor* actor) {
    if (actor)
        _groundActors.insert(actor);
}

void PhysicsWorld::unregisterGroundActor(const PxActor* actor) {
    if (actor)
        _groundActors.erase(actor);
}

bool PhysicsWorld::isGroundActor(const PxActor* actor) const {
    return actor && _groundActors.count(actor) != 0;
}

void PhysicsWorld::addBox(float x, float y, float z) {
    PxTransform boxPose(PxVec3(x, y, z));
    PxRigidDynamic* box = PxCreateDynamic(
        *_physics, boxPose, PxBoxGeometry(0.5f, 0.5f, 0.5f), *_material, 1.0f);
    _scene->addActor(*box);
}

PxRigidDynamic* PhysicsWorld::createDynamicBox(const glm::vec3& halfExtents,
                                               const glm::vec3& pos,
                                               const glm::quat& rot,
                                               float density) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidDynamic* actor = PxCreateDynamic(
        *_physics, pose,
        PxBoxGeometry(halfExtents.x, halfExtents.y, halfExtents.z), *_material,
        density);
    _scene->addActor(*actor);
    return actor;
}

PxRigidDynamic* PhysicsWorld::createDynamicSphere(float radius,
                                                  const glm::vec3& pos,
                                                  const glm::quat& rot,
                                                  float density) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidDynamic* actor = PxCreateDynamic(
        *_physics, pose, PxSphereGeometry(radius), *_material, density);
    _scene->addActor(*actor);
    return actor;
}

PxRigidDynamic*
PhysicsWorld::createDynamicRigid(const Animation::CharacterData& data,
                                 const glm::vec3& pos, const glm::quat& rot,
                                 float density, PxU32 collisionGroup,
                                 float contactOffset, float restOffset) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidDynamic* actor = _physics->createRigidDynamic(pose);
    if (!actor)
        return nullptr;

    const std::vector<Animation::CollisionGeom>* geoms = nullptr;
    auto rootIt = data.collisionGeoms.find(0);
    if (rootIt != data.collisionGeoms.end() && !rootIt->second.empty()) {
        geoms = &rootIt->second;
    } else {
        for (const auto& [bodyIdx, bodyGeoms] : data.collisionGeoms) {
            PX_UNUSED(bodyIdx);
            if (!bodyGeoms.empty()) {
                geoms = &bodyGeoms;
                break;
            }
        }
    }

    if (!geoms) {
        PxShape* shape = PxRigidActorExt::createExclusiveShape(
            *actor, PxSphereGeometry(0.1f), *_material);
        applyRigidContactOffsets(shape, contactOffset, restOffset);
    } else {
        using Type = Animation::CollisionGeom::Type;
        for (const auto& g : *geoms) {
            PxMaterial* shapeMat = _material;
            PxMaterial* ownedMat = nullptr;
            if (_physics && std::abs(g.friction - 1.f) > 1e-6f) {
                ownedMat =
                    _physics->createMaterial(g.friction, g.friction, 0.f);
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
                        *actor, PxCapsuleGeometry(radius, halfH), *shapeMat);
                    localPose = rigidFromToPose(g.from, g.to);
                } else {
                    shape = PxRigidActorExt::createExclusiveShape(
                        *actor, PxCapsuleGeometry(radius, g.size[1]),
                        *shapeMat);
                    localPose =
                        PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                                    rigidMjcfShapeRot(g.quat));
                }
                break;
            }
            case Type::Sphere:
                shape = PxRigidActorExt::createExclusiveShape(
                    *actor, PxSphereGeometry(g.size[0]), *shapeMat);
                localPose =
                    PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()));
                break;
            case Type::Box:
                shape = PxRigidActorExt::createExclusiveShape(
                    *actor, PxBoxGeometry(g.size[0], g.size[1], g.size[2]),
                    *shapeMat);
                localPose = PxTransform(
                    PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                    PxQuat(g.quat.x(), g.quat.y(), g.quat.z(), g.quat.w()));
                break;
            }

            if (shape) {
                shape->setLocalPose(localPose);
                applyRigidContactOffsets(
                    shape, g.margin >= 0.f ? g.margin : contactOffset,
                    restOffset);
            }
            if (ownedMat)
                ownedMat->release();
        }
    }

    PxRigidBodyExt::updateMassAndInertia(*actor, density);
    setRigidCollisionFilterData(actor, collisionGroup);
    _scene->addActor(*actor);
    return actor;
}

std::vector<float>
PhysicsWorld::getContactForcesFlat(const Articulation& articulation,
                                   bool groundOnly) const {
    std::vector<float> out;
    const auto& links = articulation.links();
    out.assign(links.size() * 3, 0.0f);
    if (_contacts.empty() || _dt <= 0.0f)
        return out;

    std::unordered_map<const PxActor*, std::size_t> linkIndices;
    linkIndices.reserve(links.size());
    for (std::size_t i = 0; i < links.size(); ++i) {
        if (links[i])
            linkIndices.emplace(links[i], i);
    }

    auto addForce = [&out, &linkIndices](const PxActor* actor,
                                         const glm::vec3& force) {
        const auto it = linkIndices.find(actor);
        if (it == linkIndices.end())
            return;
        const std::size_t i = it->second;
        out[i * 3 + 0] += force.x;
        out[i * 3 + 1] += force.y;
        out[i * 3 + 2] += force.z;
    };

    for (const auto& contact : _contacts) {
        const bool actor0IsGround = isGroundActor(contact.actor0);
        const bool actor1IsGround = isGroundActor(contact.actor1);
        if (groundOnly && !actor0IsGround && !actor1IsGround)
            continue;

        const glm::vec3 force0 = contact.impulse / _dt;
        const glm::vec3 force1 = -force0;
        if (!groundOnly || actor1IsGround)
            addForce(contact.actor0, force0);
        if (!groundOnly || actor0IsGround)
            addForce(contact.actor1, force1);
    }
    return out;
}

std::vector<float>
PhysicsWorld::getRigidContactForceFlat(const PxRigidDynamic& rigid,
                                       bool groundOnly) const {
    std::vector<float> out(3, 0.0f);
    if (_contacts.empty() || _dt <= 0.0f)
        return out;

    const PxActor* rigidActor = &rigid;
    auto addForce = [&out](const glm::vec3& force) {
        out[0] += force.x;
        out[1] += force.y;
        out[2] += force.z;
    };

    for (const auto& contact : _contacts) {
        const bool actor0IsRigid = contact.actor0 == rigidActor;
        const bool actor1IsRigid = contact.actor1 == rigidActor;
        if (!actor0IsRigid && !actor1IsRigid)
            continue;

        const bool actor0IsGround = isGroundActor(contact.actor0);
        const bool actor1IsGround = isGroundActor(contact.actor1);
        if (groundOnly && !actor0IsGround && !actor1IsGround)
            continue;

        const glm::vec3 force0 = contact.impulse / _dt;
        const glm::vec3 force1 = -force0;
        if (actor0IsRigid && (!groundOnly || actor1IsGround))
            addForce(force0);
        if (actor1IsRigid && (!groundOnly || actor0IsGround))
            addForce(force1);
    }

    return out;
}

} // namespace KE

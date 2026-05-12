#include "physics.hpp"
#include "articulation.hpp"
#include "foundation/Px.h"
#include <unordered_map>
#include <vector>

namespace KE {

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

} // namespace KE

#include "physics.hpp"
#include "foundation/Px.h"

namespace KE {

void PhysicsWorld::step() {
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

} // namespace KE

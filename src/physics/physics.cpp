#include "physics.hpp"

namespace KE {

void PhysicsWorld::step()
{
    mScene->simulate(mdt);  // mdt is already deltaTime (1/60)
    mScene->fetchResults(true);
}

void PhysicsWorld::fecthData()
{
    // TODO: complete me
    PxU32 nbActors = mScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);

}

void PhysicsWorld::addDefaultGround()
{
    PxVec3 normal = (mUpAxis == UpAxis::Z) ? PxVec3(0, 0, 1) : PxVec3(0, 1, 0);
    PxRigidStatic* groundPlane = PxCreatePlane(*mPhysics, PxPlane(normal, 0), *mMaterial);
    mScene->addActor(*groundPlane);
}

void PhysicsWorld::addBox(float x, float y, float z)
{
    PxTransform boxPose(PxVec3(x, y, z));
    PxRigidDynamic* box = PxCreateDynamic(*mPhysics, boxPose, PxBoxGeometry(0.5f, 0.5f, 0.5f), *mMaterial, 1.0f);
    mScene->addActor(*box);
}

} // namespace KE


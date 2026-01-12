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
    PxRigidStatic* groundPlane = PxCreatePlane(*mPhysics, PxPlane(0, 1, 0, 0), *mMaterial); // TODO: set exact up-axis
    mScene->addActor(*groundPlane);
}

void PhysicsWorld::addBox(float x, float y, float z)
{
    PxTransform boxPose(PxVec3(x, y, z));
    PxRigidDynamic* box = PxCreateDynamic(*mPhysics, boxPose, PxBoxGeometry(0.5f, 0.5f, 0.5f), *mMaterial, 1.0f);
    mScene->addActor(*box);
}

} // namespace KE


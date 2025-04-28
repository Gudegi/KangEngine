///
/// Author Kyungwon Kang, 2025/04
///

#ifndef _PHYSICS_HPP_
#define _PHYSICS_HPP_

#include "PxPhysicsAPI.h"

using namespace physx;

struct PhysicsConfig
{
    float dt;
    PxVec3 gravity; // gravity = PxVec3(0.0f, -9.81f, 0.0f);
    PxVec3 friction;
};

class PhysicsWorld
{

private:

    PxDefaultAllocator      mAllocator;
    PxDefaultErrorCallback  mErrorCallback;
    PxFoundation*           mFoundation = nullptr;
    PxPhysics*              mPhysics    = nullptr;
    PxScene*                mScene      = nullptr;
    PxMaterial*             mMaterial   = nullptr;

public:

    PhysicsWorld(PhysicsConfig config)
    {
        mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, mAllocator, mErrorCallback);
        mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mFoundation, PxTolerancesScale(), true);

        PxSceneDesc sceneDesc(mPhysics->getTolerancesScale());
        sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f); // TODO: change
        sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(2); // 2 threads
        sceneDesc.filterShader = PxDefaultSimulationFilterShader;
        mScene = mPhysics->createScene(sceneDesc);

        gMaterial = mPhysics->createMaterial(0.5f, 0.5f, 0.6f); // staticFriction, dynamicFriction, restitution
    }

    ~PhysicsWorld()
    {
        mScene->release();
        mMaterial->release();
        mPhysics->release();
        mFoundation->release();
    };

    void addDefaultGround();

    void addBox();

    void fecthData();

    void step();

};

#endif
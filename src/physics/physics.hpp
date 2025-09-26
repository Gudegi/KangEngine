///
/// Author Kyungwon Kang, 2025/04
///

#ifndef _PHYSICS_HPP_
#define _PHYSICS_HPP_

#include "PxPhysicsAPI.h"

using namespace physx;

namespace KE {

struct PhysicsConfig
{
    float dt = 1.0 / 60.0f;
    float gravity[3] = {0.0f, -9.81f, 0.0f};
    float friction[3] = {1.0f, 1.0f, 1.0f};
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
    
    float mdt;
    PxVec3 mGravity;
    PxVec3 mFriction;

public:

    PhysicsWorld(PhysicsConfig config)
    {
        mdt = config.dt;
        mGravity = PxVec3(config.gravity[0], config.gravity[1], config.gravity[2]);
        mFriction = PxVec3(config.friction[0], config.friction[1], config.friction[2]);

        mFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, mAllocator, mErrorCallback);
        mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mFoundation, PxTolerancesScale(), true);

        PxSceneDesc sceneDesc(mPhysics->getTolerancesScale());
        sceneDesc.gravity = mGravity;
        sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(2); // 2 threads
        sceneDesc.filterShader = PxDefaultSimulationFilterShader;
        mScene = mPhysics->createScene(sceneDesc);

        mMaterial = mPhysics->createMaterial(mFriction[0], mFriction[1], mFriction[2]); // staticFriction, dynamicFriction, restitution
    }

    ~PhysicsWorld()
    {
        mScene->release();
        mMaterial->release();
        mPhysics->release();
        mFoundation->release();
    };

    void setDt(float dt) {mdt = dt ;} 

    void addDefaultGround();

    void addBox(float x, float y, float z);

    void fecthData();

    void step();

};

} // namespace KE

#endif
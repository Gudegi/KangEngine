///
/// Author Kyungwon Kang, 2025/04
///

#ifndef _PHYSICS_HPP_
#define _PHYSICS_HPP_

#include "PxPhysicsAPI.h"
#include <fmt/base.h>
#include "utils/types.hpp"

using namespace physx;

namespace KE {

struct PhysicsConfig {
    UpAxis upAxis = UpAxis::Y;
    float dt = 1.0f / 60.0f;
    float gravity[3] = {0.0f, -9.81f, 0.0f};
    float friction[3] = {1.0f, 1.0f, 0.0f};
    PxSimulationFilterShader filterShader = PxDefaultSimulationFilterShader;
    PxSolverType::Enum solverType = PxSolverType::ePGS;

    static PhysicsConfig yUp() { return {}; }

    static PhysicsConfig zUp() {
        PhysicsConfig c;
        c.upAxis = UpAxis::Z;
        c.gravity[1] = 0.f;
        c.gravity[2] = -9.81f;
        return c;
    }
};

class PhysicsWorld {

  private:
    PxDefaultAllocator mAllocator;
    PxDefaultErrorCallback mErrorCallback;
    PxFoundation* mFoundation = nullptr;
    PxPhysics* mPhysics = nullptr;
    PxScene* mScene = nullptr;
    PxMaterial* mMaterial = nullptr;

    float mdt;
    UpAxis mUpAxis;
    PxVec3 mGravity;
    PxVec3 mFriction;

  public:
    PhysicsWorld(PhysicsConfig config) {
        mdt = config.dt;
        mUpAxis = config.upAxis;
        mGravity =
            PxVec3(config.gravity[0], config.gravity[1], config.gravity[2]);
        mFriction =
            PxVec3(config.friction[0], config.friction[1], config.friction[2]);

        mFoundation =
            PxCreateFoundation(PX_PHYSICS_VERSION, mAllocator, mErrorCallback);
        mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mFoundation,
                                   PxTolerancesScale(), true);
        PxInitExtensions(*mPhysics, nullptr);

        PxSceneDesc sceneDesc(mPhysics->getTolerancesScale());
        sceneDesc.gravity = mGravity;
        sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(2);
        sceneDesc.filterShader = config.filterShader;
        sceneDesc.solverType = config.solverType;
        mScene = mPhysics->createScene(sceneDesc);

        mMaterial = mPhysics->createMaterial(
            mFriction[0], mFriction[1],
            mFriction[2]); // staticFriction, dynamicFriction, restitution
        fmt::print("PhysX is initialized.\n");
    }

    ~PhysicsWorld() {
        mScene->release();
        mMaterial->release();
        PxCloseExtensions();
        mPhysics->release();
        mFoundation->release();
    };

    void setDt(float dt) { mdt = dt; }

    void addDefaultGround();

    void addBox(float x, float y, float z);

    void fecthData();

    void step();

    PxU32 numBodyActors() {
        return mScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    }

    UpAxis getUpAxis() const { return mUpAxis; }
    PxPhysics* getPhysics() { return mPhysics; }
    PxMaterial* getMaterial() { return mMaterial; }
    PxScene* getScene() { return mScene; }
};

} // namespace KE

#endif
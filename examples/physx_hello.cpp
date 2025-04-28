#include "PxPhysicsAPI.h"
#include <iostream>

using namespace physx;

PxDefaultAllocator      gAllocator;
PxDefaultErrorCallback  gErrorCallback;
PxFoundation*           gFoundation = nullptr;
PxPhysics*              gPhysics    = nullptr;
PxScene*                gScene      = nullptr;
PxMaterial*             gMaterial   = nullptr;

void initPhysics()
{
    // Foundation (메모리 관리, 에러 처리)
    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

    // Physics (물리 엔진 생성)
    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true);

    // Scene (중력 있는 월드)
    PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(2); // 2 threads
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    gScene = gPhysics->createScene(sceneDesc);

    // Material (마찰, 반발)
    gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f); // staticFriction, dynamicFriction, restitution

    // Static ground plane (y=0)
    PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
    gScene->addActor(*groundPlane);

    // Dynamic box (1m cube)
    PxTransform boxPose(PxVec3(0.0f, 10.0f, 0.0f));
    PxRigidDynamic* box = PxCreateDynamic(*gPhysics, boxPose, PxBoxGeometry(0.5f, 0.5f, 0.5f), *gMaterial, 1.0f);
    gScene->addActor(*box);
}

void stepPhysics()
{
    // 시뮬레이션 1/60초 진행
    gScene->simulate(1.0f / 60.0f);
    gScene->fetchResults(true);

    // 첫 번째 actor(box) 위치 출력
    PxU32 nbActors = gScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    if (nbActors)
    {
        PxActor* actors[1];
        gScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC, actors, 1);
        PxRigidActor* actor = static_cast<PxRigidActor*>(actors[0]);
        PxTransform pose = actor->getGlobalPose();
        std::cout << "Box position: (" 
                  << pose.p.x << ", " 
                  << pose.p.y << ", " 
                  << pose.p.z << ")" << std::endl;
    }
}

void cleanupPhysics()
{
    gScene->release();
    gMaterial->release();
    gPhysics->release();
    gFoundation->release();
}

int main()
{
    initPhysics();
    stepPhysics();
    cleanupPhysics();
    return 0;
}

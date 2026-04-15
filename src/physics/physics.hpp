///
/// Author Kyungwon Kang, 2025/04
///

#ifndef _PHYSICS_HPP_
#define _PHYSICS_HPP_

#include "PxPhysicsAPI.h"
#include <fmt/base.h>
#include "utils/types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "engine/scene/scene_backend.hpp"

using namespace physx;

namespace KE {

struct PhysicsConfig {
    UpAxis upAxis = UpAxis::Y;
    float dt = 1.0f / 60.0f;
    float gravity[3] = {0.0f, -9.81f, 0.0f};
    float friction[3] = {1.0f, 1.0f, 0.0f};
    PxSimulationFilterShader filterShader = PxDefaultSimulationFilterShader;
    // PxSolverType::Enum solverType = PxSolverType::ePGS;
    PxSolverType::Enum solverType = PxSolverType::eTGS;

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
    PxDefaultAllocator _allocator;
    PxDefaultErrorCallback _errorCallback;
    PxFoundation* _foundation = nullptr;
    PxPhysics* _physics = nullptr;
    PxScene* _scene = nullptr;
    PxMaterial* _material = nullptr;

    float _dt;
    UpAxis _upAxis;
    PxVec3 _gravity;
    PxVec3 _friction;

  public:
    PhysicsWorld(PhysicsConfig config) {
        _dt = config.dt;
        _upAxis = config.upAxis;
        _gravity =
            PxVec3(config.gravity[0], config.gravity[1], config.gravity[2]);
        _friction =
            PxVec3(config.friction[0], config.friction[1], config.friction[2]);

        _foundation =
            PxCreateFoundation(PX_PHYSICS_VERSION, _allocator, _errorCallback);
        _physics = PxCreatePhysics(PX_PHYSICS_VERSION, *_foundation,
                                   PxTolerancesScale(), true);
        PxInitExtensions(*_physics, nullptr);

        PxSceneDesc sceneDesc(_physics->getTolerancesScale());
        sceneDesc.gravity = _gravity;
        sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(2);
        sceneDesc.filterShader = config.filterShader;
        sceneDesc.solverType = config.solverType;
        _scene = _physics->createScene(sceneDesc);

        _material = _physics->createMaterial(
            _friction[0], _friction[1],
            _friction[2]); // staticFriction, dynamicFriction, restitution
        fmt::print("PhysX is initialized.\n");
    }

    ~PhysicsWorld() {
        _scene->release();
        _material->release();
        PxCloseExtensions();
        _physics->release();
        _foundation->release();
    };

    void setDt(float dt) { _dt = dt; }

    void addDefaultGround();

    void addBox(float x, float y, float z);

    void fecthData();

    void step();

    PxU32 numBodyActors() {
        return _scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    }

    UpAxis getUpAxis() const { return _upAxis; }
    PxPhysics* getPhysics() { return _physics; }
    PxMaterial* getMaterial() { return _material; }
    PxScene* getScene() { return _scene; }

    struct RigidVisual {
        PxArticulationLink* link = nullptr;
        PxRigidDynamic* rigid = nullptr;
        Scene::Prim* prim = nullptr;
    };
    std::vector<RigidVisual> _rigidVisuals;
    void registerRigidVisual(const RigidVisual& rigidVisual) {
        _rigidVisuals.push_back(rigidVisual);
    }
    void syncAllVisuals();
};

// PhysX > GLM conversion
inline glm::vec3 pxToGlm(const PxVec3& v) { return glm::vec3(v.x, v.y, v.z); }
inline glm::quat pxToGlm(const PxQuat& q) {
    return glm::quat(q.w, q.x, q.y, q.z);
}

} // namespace KE

#endif
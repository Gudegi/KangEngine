#pragma once

#include "physics/force_drag_controller.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/vec3.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace physx {
class PxRigidActor;
class PxRigidDynamic;
}

namespace KE {

class PhysicsWorld;

namespace Physics {
struct PhysicsResourceLifetimeToken;
}

namespace Scene {

class CollisionShapeComponent;
class Prim;
class RigidBodyComponent;
class SceneBackend;

// SceneGraph-only rigid-body authoring and optional CPU PhysX runtime system.
// PhysicsWorld remains caller-owned and caller-stepped. ExternalBuffer and
// articulation runtime objects are intentionally rejected to avoid duplicate
// transform and actor ownership.
class ScenePhysicsSystem {
  public:
    ScenePhysicsSystem();
    ~ScenePhysicsSystem();

    ScenePhysicsSystem(const ScenePhysicsSystem&) = delete;
    ScenePhysicsSystem& operator=(const ScenePhysicsSystem&) = delete;

    void bind(SceneBackend* scene);
    void bindPhysicsWorld(PhysicsWorld* world);
    void unbindPhysicsWorld();
    PhysicsWorld* physicsWorld() const;
    bool hasRuntimeWorld() const;
    std::shared_ptr<RigidBodyComponent> registerRigidBody(Prim& root);
    bool unregister(RigidBodyComponent& component);
    void refresh(RigidBodyComponent& component);
    void detachSubtree(Prim& root);
    void clear();

    bool isRegistered(const RigidBodyComponent& component) const;
    size_t registrationCount() const { return _registrations.size(); }
    size_t collisionShapeCount(const RigidBodyComponent& component) const;
    std::vector<std::string>
    collisionPrimPaths(const RigidBodyComponent& component) const;
    bool hasRuntimeActor(const RigidBodyComponent& component) const;
    physx::PxRigidActor*
    runtimeActor(const RigidBodyComponent& component) const;
    const std::string& runtimeError(const RigidBodyComponent& component) const;
    std::shared_ptr<RigidBodyComponent>
    registeredRigidBodyForPrim(const Prim& prim) const;
    bool beginForceDrag(const Prim& prim, const glm::vec3& hitPosition,
                        const glm::vec3& target);
    void updateForceDrag(const glm::vec3& target);
    void endForceDrag();
    bool isForceDragActive() const { return _forceDragController.active(); }
    const glm::vec3& forceDragAnchorPosition() const {
        return _forceDragController.lastAnchorPosition();
    }
    const glm::vec3& forceDragTarget() const {
        return _forceDragController.lastTarget();
    }
    const glm::vec3& forceDragForce() const {
        return _forceDragController.lastForce();
    }

    size_t contactCount(const RigidBodyComponent& component) const;

    void syncBeforeSimulation();
    void syncAfterSimulation();

  private:
    struct Registration {
        Prim* root = nullptr;
        std::vector<Prim*> collisionPrims;
        physx::PxRigidActor* actor = nullptr;
        uint64_t lastTransformVersion = 0;
        std::string runtimeError;
    };

    std::vector<Prim*> collectCollisionPrims(Prim& root) const;
    Registration& requireRegistration(const RigidBodyComponent& component);
    const Registration&
    requireRegistration(const RigidBodyComponent& component) const;
    void installCollisionCallbacks(Registration& registration,
                                   RigidBodyComponent& component);
    void clearCollisionCallbacks(Registration& registration);
    physx::PxRigidActor* createRuntimeActor(const Registration& registration,
                                            RigidBodyComponent& component);
    void destroyRuntimeActor(Registration& registration);
    void rebuildRuntimeActor(Registration& registration,
                             RigidBodyComponent& component,
                             std::vector<Prim*> collisionPrims);
    std::unordered_map<RigidBodyComponent*, Registration> _registrations;
    SceneBackend* _scene = nullptr;
    PhysicsWorld* _physicsWorld = nullptr;
    std::weak_ptr<Physics::PhysicsResourceLifetimeToken> _physicsLifetime;
    ForceDragController _forceDragController;
    physx::PxRigidDynamic* _forceDragActor = nullptr;
};

} // namespace Scene
} // namespace KE

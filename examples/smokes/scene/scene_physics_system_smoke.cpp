#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/scene/component/collision_shape_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/rigid_body_component.hpp"
#include "engine/scene/component/scene_physics_system.hpp"
#include "engine/scene/native/native_scene.hpp"
#include "engine/scene/native/prim.hpp"
#include "physics/physics.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace KE;
using namespace KE::Scene;

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Fn> void requireThrows(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    NativeScene scene;
    ScenePhysicsSystem physicsSystem;
    physicsSystem.bind(&scene);

    Prim* body = scene.definePrim("/Body", PrimType::Xform);
    body->setWorldTranslation({0.0f, 2.0f, 0.0f});
    auto rigid = body->addRigidBodyComponent();
    rigid->setBodyType(RigidBodyType::Dynamic);
    rigid->setDensity(750.0f);
    require(rigid->transformMode() == PhysicsTransformMode::PhysicsToScene,
            "dynamic rigid body transform ownership is incorrect");
    rigid->setContactOffsets(0.03f, -0.002f);
    require(rigid->contactOffset() == 0.03f &&
                rigid->restOffset() == -0.002f,
            "rigid body contact offsets were not stored");
    requireThrows([&] { rigid->setContactOffsets(0.0f, 0.0f); },
                  "non-positive contact offset was accepted");
    requireThrows([&] { rigid->setContactOffsets(0.01f, 0.01f); },
                  "contact offset equal to rest offset was accepted");

    Prim* collider = scene.definePrim("/Body/Collision", PrimType::Mesh);
    collider->setLocalTranslation({0.25f, 0.0f, 0.0f});
    auto shape = collider->addCollisionShapeComponent();
    shape->setShapeType(CollisionShapeType::Box);
    shape->setSize({0.5f, 0.5f, 0.5f});

    physicsSystem.registerRigidBody(*body);
    require(physicsSystem.registrationCount() == 1,
            "rigid body registration failed");
    require(physicsSystem.collisionShapeCount(*rigid) == 1,
            "collision shape collection failed");
    const auto colliderPaths = physicsSystem.collisionPrimPaths(*rigid);
    require(colliderPaths.size() == 1 &&
                colliderPaths.front() == "/Body/Collision",
            "collision Prim path diagnostics are incorrect");

    PhysicsConfig config = PhysicsConfig::yUp();
    config.cpuDispatcherThreads = 1;
    PhysicsWorld world(config);
    world.addDefaultGround();
    physicsSystem.bindPhysicsWorld(&world);
    require(physicsSystem.hasRuntimeWorld(),
            "PhysicsWorld runtime binding failed");
    require(physicsSystem.hasRuntimeActor(*rigid),
            "registered rigid body did not create a runtime actor");
    physx::PxShape* runtimeShape = nullptr;
    physicsSystem.runtimeActor(*rigid)->getShapes(&runtimeShape, 1);
    require(runtimeShape &&
                std::abs(runtimeShape->getLocalPose().p.x - 0.25f) < 1e-4f,
            "collision Prim body-local pose was not used as the shape pose");
    const float initialHeight = body->getWorldTranslation().y;
    world.step();
    physicsSystem.syncAfterSimulation();
    require(body->getWorldTranslation().y < initialHeight,
            "dynamic Physics-to-Scene synchronization failed");
    auto* dynamicActor = static_cast<physx::PxRigidDynamic*>(
        physicsSystem.runtimeActor(*rigid));
    const glm::vec3 dragStart = body->getWorldTranslation();
    require(physicsSystem.beginForceDrag(
                *body, dragStart,
                dragStart + glm::vec3(1.0f, 0.0f, 0.0f)),
            "Scene rigid force drag did not begin");
    require(std::abs(glm::length(physicsSystem.forceDragForce()) - 300.0f) <
                1e-3f,
            "Scene rigid force drag did not use the configured force clamp");
    physicsSystem.syncBeforeSimulation();
    world.step();
    require(dynamicActor->getLinearVelocity().x > 0.0f,
            "Scene rigid force drag did not apply force");
    physicsSystem.endForceDrag();
    bool sawContact = false;
    for (int i = 0; i < 120; ++i) {
        world.step();
        physicsSystem.syncAfterSimulation();
        sawContact = sawContact || physicsSystem.contactCount(*rigid) > 0;
    }
    require(sawContact,
            "runtime actor contact mapping failed");
    rigid->setCollisionGroup(3);
    require(physicsSystem.isRegistered(*rigid),
            "component change invalidated registration");

    rigid->setBodyType(RigidBodyType::Kinematic);
    require(rigid->transformMode() == PhysicsTransformMode::SceneToPhysics,
            "kinematic rigid body transform ownership is incorrect");
    body->setWorldTranslation({0.0f, 3.0f, 0.0f});
    physicsSystem.syncBeforeSimulation();
    world.step();
    auto* kinematicActor = static_cast<physx::PxRigidDynamic*>(
        physicsSystem.runtimeActor(*rigid));
    require(kinematicActor && kinematicActor->getGlobalPose().p.y > 2.9f,
            "kinematic Scene-to-Physics synchronization failed");

    collider->removeCollisionShapeComponent();
    require(physicsSystem.registrationCount() == 0,
            "collision component detach left a stale registration");
    require(world.numBodyActors() == 0,
            "collision component detach left a runtime actor");
    shape = collider->addCollisionShapeComponent();
    shape->setShapeType(CollisionShapeType::Box);
    shape->setSize({0.5f, 0.5f, 0.5f});
    physicsSystem.registerRigidBody(*body);

    physicsSystem.detachSubtree(*collider);
    require(physicsSystem.registrationCount() == 0,
            "collider subtree detach left a stale registration");

    physicsSystem.registerRigidBody(*body);
    body->removeRigidBodyComponent();
    require(physicsSystem.registrationCount() == 0,
            "component detach did not unregister the rigid body");

    Physics::ConvexMeshPart tetrahedron;
    tetrahedron.vertices = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    };
    auto convexResource =
        world.createConvexCollision({tetrahedron, tetrahedron});
    Prim* convexBody = scene.definePrim("/ConvexBody", PrimType::Xform);
    auto convexRigid = convexBody->addRigidBodyComponent();
    convexRigid->setBodyType(RigidBodyType::Dynamic);
    auto convexShape = convexBody->addCollisionShapeComponent();
    convexShape->setConvexResource(convexResource);
    physicsSystem.registerRigidBody(*convexBody);
    auto* convexActor = physicsSystem.runtimeActor(*convexRigid);
    require(convexActor && convexActor->getNbShapes() == 2,
            "convex compound resource did not create one shape per part");
    convexBody->removeRigidBodyComponent();

    Prim* staticBody = scene.definePrim("/StaticBody", PrimType::Xform);
    auto staticRigid = staticBody->addRigidBodyComponent();
    staticRigid->setBodyType(RigidBodyType::Static);
    auto staticShape = staticBody->addCollisionShapeComponent();
    staticShape->setShapeType(CollisionShapeType::Sphere);
    staticShape->setSize({0.25f, 0.0f, 0.0f});
    physicsSystem.registerRigidBody(*staticBody);
    physx::PxShape* rootShape = nullptr;
    physicsSystem.runtimeActor(*staticRigid)->getShapes(&rootShape, 1);
    require(rootShape && rootShape->getLocalPose().isSane() &&
                rootShape->getLocalPose().p.isZero() &&
                rootShape->getLocalPose().q.isIdentity(),
            "a collision shape on the rigid body Prim must resolve to an "
            "identity actor-local pose");
    staticBody->setWorldTranslation({2.0f, 0.25f, 0.0f});
    physicsSystem.syncBeforeSimulation();
    require(std::abs(physicsSystem.runtimeActor(*staticRigid)
                         ->getGlobalPose()
                         .p.x -
                     2.0f) < 1e-4f,
            "static Scene-to-Physics synchronization failed");
    staticBody->setVisible(false);
    physicsSystem.syncBeforeSimulation();
    require(!physicsSystem.runtimeActor(*staticRigid)
                 ->getActorFlags()
                 .isSet(physx::PxActorFlag::eDISABLE_SIMULATION),
            "render visibility incorrectly disabled physics");
    staticBody->setActive(false);
    physicsSystem.syncBeforeSimulation();
    require(physicsSystem.runtimeActor(*staticRigid)
                ->getActorFlags()
                .isSet(physx::PxActorFlag::eDISABLE_SIMULATION),
            "inactive Prim did not disable physics participation");
    staticBody->removeRigidBodyComponent();

    Prim* external = scene.definePrim("/External", PrimType::Mesh);
    external->addRigidBodyComponent();
    external->addCollisionShapeComponent();
    auto render = external->addRenderComponent();
    render->setTransformSource(TransformSource::ExternalBuffer);
    requireThrows(
        [&] { physicsSystem.registerRigidBody(*external); },
        "ExternalBuffer renderable was accepted by ScenePhysicsSystem");

    Prim* articulation = scene.definePrim("/Articulation", PrimType::Xform);
    articulation->addRigidBodyComponent();
    articulation->addCollisionShapeComponent();
    articulation->addArticulationComponent();
    requireThrows([&] { physicsSystem.registerRigidBody(*articulation); },
                  "articulation subtree was accepted by ScenePhysicsSystem");

    std::cout << "scene_physics_system_smoke: PASS\n";
    return 0;
}

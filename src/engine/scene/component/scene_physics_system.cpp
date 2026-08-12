#include "engine/scene/component/scene_physics_system.hpp"

#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/scene/component/collision_shape_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/rigid_body_component.hpp"
#include "engine/scene/component/transform_component.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"

#ifdef KANGENGINE_USE_PHYSX
#include "physics/collision/convex_collision.hpp"
#include "physics/physics.hpp"
#include "physics/physics_material.hpp"
#include <glm/gtx/quaternion.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace KE {
namespace Scene {

namespace {
#ifdef KANGENGINE_USE_PHYSX
physx::PxTransform toPxTransform(const glm::vec3& position,
                                 glm::quat rotation) {
    if (!std::isfinite(rotation.x) || !std::isfinite(rotation.y) ||
        !std::isfinite(rotation.z) || !std::isfinite(rotation.w) ||
        glm::length(rotation) < 1e-6f) {
        throw std::invalid_argument(
            "collision rotation must be finite and non-zero");
    }
    rotation = glm::normalize(rotation);
    return physx::PxTransform(
        physx::PxVec3(position.x, position.y, position.z),
        physx::PxQuat(rotation.x, rotation.y, rotation.z, rotation.w));
}

physx::PxTransform collisionPrimBodyPose(Prim& bodyRoot,
                                         Prim& collisionPrim) {
    if (&bodyRoot == &collisionPrim)
        return physx::PxTransform(physx::PxIdentity);

    const glm::quat bodyRotation =
        glm::normalize(bodyRoot.getWorldRotation());
    const glm::quat collisionRotation =
        glm::normalize(collisionPrim.getWorldRotation());
    const glm::quat bodyLocalRotation =
        glm::normalize(glm::inverse(bodyRotation) * collisionRotation);
    const glm::vec3 bodyLocalPosition =
        glm::inverse(bodyRotation) *
        (collisionPrim.getWorldTranslation() -
         bodyRoot.getWorldTranslation());
    return toPxTransform(bodyLocalPosition, bodyLocalRotation);
}

void validatePositive(float value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0f)
        throw std::invalid_argument(std::string(name) +
                                    " must be finite and positive");
}
#endif
} // namespace

ScenePhysicsSystem::ScenePhysicsSystem()
    : _forceDragController(ForceDragConfig{750.0f, 75.0f, 300.0f}) {}

ScenePhysicsSystem::~ScenePhysicsSystem() { clear(); }

void ScenePhysicsSystem::bind(SceneBackend* scene) {
    if (!_registrations.empty() && scene != _scene)
        throw std::runtime_error(
            "cannot rebind ScenePhysicsSystem with active registrations");
    _scene = scene;
}

void ScenePhysicsSystem::bindPhysicsWorld(PhysicsWorld* world) {
    if (world == physicsWorld())
        return;
    unbindPhysicsWorld();
    if (!world)
        return;
#ifdef KANGENGINE_USE_PHYSX
    if (world->isGpuEnabled())
        throw std::runtime_error(
            "ScenePhysicsSystem currently supports only CPU PhysicsWorld");
    _physicsWorld = world;
    _physicsLifetime = world->lifetimeToken();
    try {
        for (auto& [component, registration] : _registrations) {
            if (component)
                registration.actor =
                    createRuntimeActor(registration, *component);
        }
    } catch (...) {
        unbindPhysicsWorld();
        throw;
    }
#else
    (void)world;
    throw std::runtime_error(
        "ScenePhysicsSystem runtime requires a PhysX build");
#endif
}

void ScenePhysicsSystem::unbindPhysicsWorld() {
    const bool worldAlive = hasRuntimeWorld();
    for (auto& [component, registration] : _registrations) {
        (void)component;
        if (worldAlive)
            destroyRuntimeActor(registration);
        else
            registration.actor = nullptr;
    }
    _physicsWorld = nullptr;
    _physicsLifetime.reset();
}

PhysicsWorld* ScenePhysicsSystem::physicsWorld() const {
    return hasRuntimeWorld() ? _physicsWorld : nullptr;
}

bool ScenePhysicsSystem::hasRuntimeWorld() const {
    return _physicsWorld && !_physicsLifetime.expired();
}

std::vector<Prim*> ScenePhysicsSystem::collectCollisionPrims(Prim& root) const {
    std::vector<Prim*> collisionPrims;

    const auto visit = [&](const auto& self, Prim& prim, bool isRoot) -> void {
        if (!isRoot && prim.hasRigidBodyComponent())
            return;
        if (prim.hasArticulationComponent() ||
            prim.hasArticulationBindingComponent()) {
            throw std::runtime_error(
                "ScenePhysicsSystem does not accept articulation subtrees");
        }
        if (auto render = prim.getRenderComponent();
            render &&
            render->transformSource() == TransformSource::ExternalBuffer) {
            throw std::runtime_error("ScenePhysicsSystem does not accept "
                                     "ExternalBuffer renderables");
        }
        if (prim.hasCollisionShapeComponent())
            collisionPrims.push_back(&prim);
        for (Prim* child : prim.getChildren()) {
            if (child)
                self(self, *child, false);
        }
    };

    visit(visit, root, true);
    if (collisionPrims.empty()) {
        throw std::runtime_error(
            "ScenePhysicsSystem requires at least one CollisionShapeComponent "
            "under the rigid body root");
    }
    return collisionPrims;
}

std::shared_ptr<RigidBodyComponent>
ScenePhysicsSystem::registerRigidBody(Prim& root) {
    if (!_scene)
        throw std::runtime_error(
            "ScenePhysicsSystem is not bound to a Native Scene");
    if (_scene->getBackendType() != BackendType::Native)
        throw std::runtime_error(
            "ScenePhysicsSystem currently supports only Native Scene");
    if (_scene->getPrimAtPath(root.getPath()) != &root)
        throw std::runtime_error(
            "ScenePhysicsSystem cannot register a Prim from another Scene");
    auto component = root.getRigidBodyComponent();
    if (!component || !component->isAttached()) {
        throw std::runtime_error(
            "ScenePhysicsSystem requires an attached RigidBodyComponent");
    }
    if (isRegistered(*component))
        throw std::runtime_error("RigidBodyComponent is already registered");

    Registration registration;
    registration.root = &root;
    registration.collisionPrims = collectCollisionPrims(root);
    _registrations.emplace(component.get(), std::move(registration));
    component->setRegistrationCallbacks(
        [this](RigidBodyComponent& detached) { unregister(detached); },
        [this](RigidBodyComponent& changed) {
            try {
                refresh(changed);
            } catch (const std::exception& error) {
                if (isRegistered(changed))
                    requireRegistration(changed).runtimeError = error.what();
            }
        });
    Registration& registered = _registrations.at(component.get());
    installCollisionCallbacks(registered, *component);
    try {
        if (hasRuntimeWorld())
            registered.actor = createRuntimeActor(registered, *component);
    } catch (...) {
        unregister(*component);
        throw;
    }
    return component;
}

bool ScenePhysicsSystem::unregister(RigidBodyComponent& component) {
    auto it = _registrations.find(&component);
    if (it == _registrations.end())
        return false;
    Registration registration = std::move(it->second);
    _registrations.erase(it);
    component.clearRegistrationCallbacks();
    clearCollisionCallbacks(registration);
    destroyRuntimeActor(registration);
    return true;
}

void ScenePhysicsSystem::refresh(RigidBodyComponent& component) {
    auto& registration = requireRegistration(component);
    if (!registration.root)
        throw std::runtime_error("registered rigid body has no root Prim");
    std::vector<Prim*> collisionPrims =
        collectCollisionPrims(*registration.root);
    rebuildRuntimeActor(registration, component, std::move(collisionPrims));
}

void ScenePhysicsSystem::installCollisionCallbacks(
    Registration& registration, RigidBodyComponent& component) {
    for (Prim* collisionPrim : registration.collisionPrims) {
        if (auto collision = collisionPrim->getCollisionShapeComponent()) {
            collision->setRegistrationCallbacks(
                [this, componentPtr = &component](CollisionShapeComponent&) {
                    if (componentPtr && isRegistered(*componentPtr))
                        unregister(*componentPtr);
                },
                [this, componentPtr = &component](CollisionShapeComponent&) {
                    if (!componentPtr || !isRegistered(*componentPtr))
                        return;
                    try {
                        refresh(*componentPtr);
                    } catch (const std::exception& error) {
                        if (isRegistered(*componentPtr)) {
                            requireRegistration(*componentPtr).runtimeError =
                                error.what();
                        }
                    }
                });
        }
    }
}

void ScenePhysicsSystem::clearCollisionCallbacks(Registration& registration) {
    for (Prim* collisionPrim : registration.collisionPrims) {
        if (!collisionPrim)
            continue;
        if (auto collision = collisionPrim->getCollisionShapeComponent())
            collision->clearRegistrationCallbacks();
    }
}

void ScenePhysicsSystem::rebuildRuntimeActor(
    Registration& registration, RigidBodyComponent& component,
    std::vector<Prim*> collisionPrims) {
    Registration candidate = registration;
    candidate.collisionPrims = std::move(collisionPrims);
    candidate.actor = nullptr;
    if (hasRuntimeWorld())
        candidate.actor = createRuntimeActor(candidate, component);

    clearCollisionCallbacks(registration);
    destroyRuntimeActor(registration);
    registration.collisionPrims = std::move(candidate.collisionPrims);
    registration.actor = candidate.actor;
    registration.lastTransformVersion = candidate.lastTransformVersion;
    registration.runtimeError.clear();
    installCollisionCallbacks(registration, component);
}

void ScenePhysicsSystem::detachSubtree(Prim& root) {
    std::unordered_set<Prim*> subtree;
    root.traverse([&subtree](Prim* prim) {
        if (prim)
            subtree.insert(prim);
    });

    std::vector<RigidBodyComponent*> components;
    for (const auto& [component, registration] : _registrations) {
        bool intersects =
            registration.root && subtree.count(registration.root) != 0;
        if (!intersects) {
            for (Prim* collisionPrim : registration.collisionPrims) {
                if (collisionPrim && subtree.count(collisionPrim) != 0) {
                    intersects = true;
                    break;
                }
            }
        }
        if (intersects)
            components.push_back(component);
    }
    for (RigidBodyComponent* component : components) {
        if (component)
            unregister(*component);
    }
}

void ScenePhysicsSystem::clear() {
    while (!_registrations.empty())
        unregister(*_registrations.begin()->first);
}

bool ScenePhysicsSystem::isRegistered(
    const RigidBodyComponent& component) const {
    return _registrations.find(const_cast<RigidBodyComponent*>(&component)) !=
           _registrations.end();
}

size_t ScenePhysicsSystem::collisionShapeCount(
    const RigidBodyComponent& component) const {
    return requireRegistration(component).collisionPrims.size();
}

std::vector<std::string> ScenePhysicsSystem::collisionPrimPaths(
    const RigidBodyComponent& component) const {
    const Registration& registration = requireRegistration(component);
    std::vector<std::string> paths;
    paths.reserve(registration.collisionPrims.size());
    for (const Prim* prim : registration.collisionPrims) {
        if (prim)
            paths.push_back(prim->getPath());
    }
    return paths;
}

bool ScenePhysicsSystem::hasRuntimeActor(
    const RigidBodyComponent& component) const {
    return requireRegistration(component).actor != nullptr && hasRuntimeWorld();
}

physx::PxRigidActor*
ScenePhysicsSystem::runtimeActor(const RigidBodyComponent& component) const {
    const Registration& registration = requireRegistration(component);
    return hasRuntimeWorld() ? registration.actor : nullptr;
}

const std::string&
ScenePhysicsSystem::runtimeError(const RigidBodyComponent& component) const {
    return requireRegistration(component).runtimeError;
}

std::shared_ptr<RigidBodyComponent>
ScenePhysicsSystem::registeredRigidBodyForPrim(const Prim& prim) const {
    for (const Prim* candidate = &prim; candidate;
         candidate = candidate->getParent()) {
        auto component = candidate->getRigidBodyComponent();
        if (component && isRegistered(*component))
            return component;
    }
    return nullptr;
}

bool ScenePhysicsSystem::beginForceDrag(const Prim& prim,
                                        const glm::vec3& hitPosition,
                                        const glm::vec3& target) {
    endForceDrag();
#ifdef KANGENGINE_USE_PHYSX
    auto component = registeredRigidBodyForPrim(prim);
    if (!component || component->bodyType() != RigidBodyType::Dynamic ||
        !component->isEnabled()) {
        return false;
    }
    auto* actor = static_cast<physx::PxRigidDynamic*>(runtimeActor(*component));
    if (!actor)
        return false;
    _forceDragActor = actor;
    _forceDragController.beginDirect(*actor, hitPosition);
    _forceDragController.computeForce(target);
    return true;
#else
    (void)prim;
    (void)hitPosition;
    (void)target;
    return false;
#endif
}

void ScenePhysicsSystem::updateForceDrag(const glm::vec3& target) {
#ifdef KANGENGINE_USE_PHYSX
    if (!_forceDragController.active() || !_forceDragActor ||
        !hasRuntimeWorld())
        return;
    _forceDragController.computeForce(target);
#else
    (void)target;
#endif
}

void ScenePhysicsSystem::endForceDrag() {
    _forceDragController.end();
    _forceDragActor = nullptr;
}

size_t
ScenePhysicsSystem::contactCount(const RigidBodyComponent& component) const {
#ifdef KANGENGINE_USE_PHYSX
    const physx::PxRigidActor* actor = runtimeActor(component);
    if (!actor || !hasRuntimeWorld())
        return 0;
    size_t count = 0;
    for (const ContactPoint& contact : _physicsWorld->getContacts()) {
        if (contact.actor0 == actor || contact.actor1 == actor)
            ++count;
    }
    return count;
#else
    (void)component;
    return 0;
#endif
}

physx::PxRigidActor*
ScenePhysicsSystem::createRuntimeActor(const Registration& registration,
                                       RigidBodyComponent& component) {
#ifdef KANGENGINE_USE_PHYSX
    if (!hasRuntimeWorld() || !registration.root)
        return nullptr;

    using namespace physx;
    const glm::vec3 position = registration.root->getWorldTranslation();
    const glm::quat rotation = registration.root->getWorldRotation();
    const PxTransform pose = toPxTransform(position, rotation);

    PxRigidActor* actor = nullptr;
    PxRigidDynamic* dynamic = nullptr;
    if (component.bodyType() == RigidBodyType::Static) {
        actor = _physicsWorld->getPhysics()->createRigidStatic(pose);
    } else {
        dynamic = _physicsWorld->getPhysics()->createRigidDynamic(pose);
        actor = dynamic;
        if (dynamic && component.bodyType() == RigidBodyType::Kinematic) {
            dynamic->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        }
    }
    if (!actor)
        throw std::runtime_error("PhysX failed to create Scene rigid actor");

    try {
        for (Prim* collisionPrim : registration.collisionPrims) {
            auto shapeComponent =
                collisionPrim ? collisionPrim->getCollisionShapeComponent()
                              : nullptr;
            if (!shapeComponent || !shapeComponent->isAttached())
                throw std::runtime_error(
                    "registered collider lost its CollisionShapeComponent");

            const Physics::PhysicsMaterialDesc material{
                shapeComponent->staticFriction(),
                shapeComponent->dynamicFriction(),
                shapeComponent->restitution()};
            const float contactOffset = shapeComponent->margin() >= 0.0f
                                            ? shapeComponent->margin()
                                            : component.contactOffset();
            const float restOffset = component.restOffset();
            const PxTransform shapePose =
                collisionPrimBodyPose(*registration.root, *collisionPrim);

            if (shapeComponent->shapeType() == CollisionShapeType::ConvexMesh) {
                auto resource = shapeComponent->convexResource();
                if (!resource)
                    throw std::runtime_error(
                        "ConvexMesh collider has no convex resource");
                if (!_physicsWorld->attachConvexCollision(
                        *actor, *resource, material,
                        glm::vec3(shapePose.p.x, shapePose.p.y, shapePose.p.z),
                        glm::quat(shapePose.q.w, shapePose.q.x, shapePose.q.y,
                                  shapePose.q.z),
                        contactOffset, restOffset)) {
                    throw std::runtime_error(
                        "failed to attach convex collision resource");
                }
                continue;
            }

            PxShape* shape = nullptr;
            const glm::vec3 size = shapeComponent->size();
            switch (shapeComponent->shapeType()) {
            case CollisionShapeType::Sphere:
                validatePositive(size.x, "sphere radius");
                shape = _physicsWorld->createExclusiveShape(
                    *actor, PxSphereGeometry(size.x), material);
                break;
            case CollisionShapeType::Capsule:
            case CollisionShapeType::Cylinder: {
                validatePositive(size.x, "capsule radius");
                validatePositive(size.y, "capsule half-height");
                // PhysX has no native cylinder geometry. Match the existing
                // MJCF rigid path and use a capsule approximation for now.
                shape = _physicsWorld->createExclusiveShape(
                    *actor, PxCapsuleGeometry(size.x, size.y), material);
                break;
            }
            case CollisionShapeType::Box:
                validatePositive(size.x, "box half extent x");
                validatePositive(size.y, "box half extent y");
                validatePositive(size.z, "box half extent z");
                shape = _physicsWorld->createExclusiveShape(
                    *actor, PxBoxGeometry(size.x, size.y, size.z), material);
                break;
            case CollisionShapeType::ConvexMesh:
                break;
            }
            if (!shape)
                throw std::runtime_error(
                    "PhysX failed to create collision shape");
            shape->setLocalPose(shapePose);
            shape->setRestOffset(restOffset);
            shape->setContactOffset(
                std::max(contactOffset, restOffset + 1e-4f));
        }

        if (dynamic)
            PxRigidBodyExt::updateMassAndInertia(*dynamic, component.density());
        _physicsWorld->setRigidCollisionGroup(*actor,
                                              component.collisionGroup());
        actor->setActorFlag(PxActorFlag::eDISABLE_SIMULATION,
                            !component.isEnabled() ||
                                !registration.root->isActiveInHierarchy());
        actor->userData = registration.root;
        _physicsWorld->addRigidActor(*actor);
        return actor;
    } catch (...) {
        actor->release();
        throw;
    }
#else
    (void)registration;
    (void)component;
    return nullptr;
#endif
}

void ScenePhysicsSystem::destroyRuntimeActor(Registration& registration) {
#ifdef KANGENGINE_USE_PHYSX
    if (_forceDragActor == registration.actor)
        endForceDrag();
    if (registration.actor && hasRuntimeWorld())
        _physicsWorld->destroyRigidActor(registration.actor);
#endif
    registration.actor = nullptr;
}

void ScenePhysicsSystem::syncBeforeSimulation() {
#ifdef KANGENGINE_USE_PHYSX
    if (!hasRuntimeWorld()) {
        if (_physicsWorld) {
            for (auto& [component, registration] : _registrations) {
                (void)component;
                registration.actor = nullptr;
            }
            _physicsWorld = nullptr;
            _physicsLifetime.reset();
        }
        return;
    }

    for (auto& [component, registration] : _registrations) {
        if (!component || !registration.actor || !registration.root)
            continue;
        const bool enabled =
            component->isEnabled() && registration.root->isActiveInHierarchy();
        registration.actor->setActorFlag(
            physx::PxActorFlag::eDISABLE_SIMULATION, !enabled);
        if (!enabled ||
            component->transformMode() != PhysicsTransformMode::SceneToPhysics)
            continue;

        auto transform = registration.root->getTransformComponent();
        const uint64_t version = transform ? transform->version() : 0;
        if (version == registration.lastTransformVersion)
            continue;
        const physx::PxTransform pose =
            toPxTransform(registration.root->getWorldTranslation(),
                          registration.root->getWorldRotation());
        if (component->bodyType() == RigidBodyType::Kinematic) {
            static_cast<physx::PxRigidDynamic*>(registration.actor)
                ->setKinematicTarget(pose);
        } else {
            registration.actor->setGlobalPose(pose);
        }
        registration.lastTransformVersion = version;
    }
    // Match the high-level simulation drag path: updateForceDrag() computes
    // one spring-damper force at the input/render rate, then each PhysX
    // substep reapplies that cached force until the next target update.
    // PhysX clears accumulated forces after every simulate().
    if (_forceDragController.active()) {
        const auto registration = std::find_if(
            _registrations.begin(), _registrations.end(),
            [this](const auto& entry) {
                return entry.second.actor == _forceDragActor;
            });
        if (registration != _registrations.end() && registration->first &&
            registration->first->isEnabled() && registration->second.root &&
            registration->second.root->isActiveInHierarchy()) {
            _forceDragController.applyCachedForce();
        }
    }
#endif
}

void ScenePhysicsSystem::syncAfterSimulation() {
#ifdef KANGENGINE_USE_PHYSX
    if (!hasRuntimeWorld())
        return;
    for (auto& [component, registration] : _registrations) {
        if (!component || !registration.actor || !registration.root ||
            component->transformMode() !=
                PhysicsTransformMode::PhysicsToScene ||
            !component->isEnabled() ||
            !registration.root->isActiveInHierarchy()) {
            continue;
        }
        const physx::PxTransform pose = registration.actor->getGlobalPose();
        registration.root->setWorldMatrix(pxToMat4(pose));
        auto transform = registration.root->getTransformComponent();
        registration.lastTransformVersion =
            transform ? transform->version() : 0;
    }
#endif
}

ScenePhysicsSystem::Registration&
ScenePhysicsSystem::requireRegistration(const RigidBodyComponent& component) {
    auto it = _registrations.find(const_cast<RigidBodyComponent*>(&component));
    if (it == _registrations.end())
        throw std::runtime_error("RigidBodyComponent is not registered");
    return it->second;
}

const ScenePhysicsSystem::Registration& ScenePhysicsSystem::requireRegistration(
    const RigidBodyComponent& component) const {
    auto it = _registrations.find(const_cast<RigidBodyComponent*>(&component));
    if (it == _registrations.end())
        throw std::runtime_error("RigidBodyComponent is not registered");
    return it->second;
}

} // namespace Scene
} // namespace KE

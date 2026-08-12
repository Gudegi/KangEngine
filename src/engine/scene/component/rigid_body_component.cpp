#include "engine/scene/component/rigid_body_component.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace KE {
namespace Scene {

RigidBodyComponent::RigidBodyComponent(Prim* owner)
    : ComponentBase(owner, "RigidBodyComponent") {}

void RigidBodyComponent::detach() {
    if (!_owner)
        return;
    auto callback = std::move(_detachCallback);
    if (callback)
        callback(*this);
    detachBase();
}

void RigidBodyComponent::setRegistrationCallbacks(
    std::function<void(RigidBodyComponent&)> detachCallback,
    std::function<void(RigidBodyComponent&)> changeCallback) {
    _detachCallback = std::move(detachCallback);
    _changeCallback = std::move(changeCallback);
}

void RigidBodyComponent::clearRegistrationCallbacks() {
    _detachCallback = {};
    _changeCallback = {};
}

void RigidBodyComponent::markChanged() {
    ComponentBase::markChanged();
    if (_changeCallback)
        _changeCallback(*this);
}

RigidBodyType RigidBodyComponent::bodyType() const {
    requireAttached();
    return _bodyType;
}

void RigidBodyComponent::setBodyType(RigidBodyType type) {
    requireAttached();
    if (_bodyType == type)
        return;
    _bodyType = type;
    markChanged();
}

PhysicsTransformMode RigidBodyComponent::transformMode() const {
    requireAttached();
    return _bodyType == RigidBodyType::Dynamic
               ? PhysicsTransformMode::PhysicsToScene
               : PhysicsTransformMode::SceneToPhysics;
}

float RigidBodyComponent::density() const {
    requireAttached();
    return _density;
}

void RigidBodyComponent::setDensity(float density) {
    requireAttached();
    if (!std::isfinite(density) || density <= 0.0f)
        throw std::invalid_argument(
            "rigid body density must be finite and positive");
    if (_density == density)
        return;
    _density = density;
    markChanged();
}

uint32_t RigidBodyComponent::collisionGroup() const {
    requireAttached();
    return _collisionGroup;
}

void RigidBodyComponent::setCollisionGroup(uint32_t group) {
    requireAttached();
    if (_collisionGroup == group)
        return;
    _collisionGroup = group;
    markChanged();
}

float RigidBodyComponent::contactOffset() const {
    requireAttached();
    return _contactOffset;
}

void RigidBodyComponent::setContactOffset(float offset) {
    setContactOffsets(offset, _restOffset);
}

float RigidBodyComponent::restOffset() const {
    requireAttached();
    return _restOffset;
}

void RigidBodyComponent::setRestOffset(float offset) {
    setContactOffsets(_contactOffset, offset);
}

void RigidBodyComponent::setContactOffsets(float contactOffset,
                                           float restOffset) {
    requireAttached();
    if (!std::isfinite(contactOffset) || !std::isfinite(restOffset))
        throw std::invalid_argument(
            "rigid body contact offsets must be finite");
    if (contactOffset <= 0.0f)
        throw std::invalid_argument(
            "rigid body contact_offset must be positive");
    if (contactOffset <= restOffset)
        throw std::invalid_argument(
            "rigid body contact_offset must be greater than rest_offset");
    if (_contactOffset == contactOffset && _restOffset == restOffset)
        return;
    _contactOffset = contactOffset;
    _restOffset = restOffset;
    markChanged();
}

bool RigidBodyComponent::isEnabled() const {
    requireAttached();
    return _enabled;
}

void RigidBodyComponent::setEnabled(bool enabled) {
    requireAttached();
    if (_enabled == enabled)
        return;
    _enabled = enabled;
    markChanged();
}

const char* rigidBodyTypeLabel(RigidBodyType type) {
    switch (type) {
    case RigidBodyType::Static:
        return "Static";
    case RigidBodyType::Dynamic:
        return "Dynamic";
    case RigidBodyType::Kinematic:
        return "Kinematic";
    }
    return "Unknown";
}

const char* physicsTransformModeLabel(PhysicsTransformMode mode) {
    switch (mode) {
    case PhysicsTransformMode::PhysicsToScene:
        return "Physics to Scene";
    case PhysicsTransformMode::SceneToPhysics:
        return "Scene to Physics";
    }
    return "Unknown";
}

} // namespace Scene
} // namespace KE

#include "engine/scene/component/collision_shape_component.hpp"

#include <utility>

namespace KE {
namespace Scene {

CollisionShapeComponent::CollisionShapeComponent(Prim* owner)
    : ComponentBase(owner, "CollisionShapeComponent") {}

void CollisionShapeComponent::detach() {
    if (!_owner)
        return;
    auto callback = std::move(_registrationDetachCallback);
    if (callback)
        callback(*this);
    detachBase();
}

void CollisionShapeComponent::setRegistrationCallbacks(
    std::function<void(CollisionShapeComponent&)> detachCallback,
    std::function<void(CollisionShapeComponent&)> changeCallback) {
    _registrationDetachCallback = std::move(detachCallback);
    _registrationChangeCallback = std::move(changeCallback);
}

void CollisionShapeComponent::clearRegistrationCallbacks() {
    _registrationDetachCallback = {};
    _registrationChangeCallback = {};
}

void CollisionShapeComponent::markChanged() {
    ComponentBase::markChanged();
    const auto callback = _registrationChangeCallback;
    if (callback)
        callback(*this);
}

void CollisionShapeComponent::setShapeType(CollisionShapeType shapeType) {
    requireAttached();
    const bool clearConvex = shapeType != CollisionShapeType::ConvexMesh &&
                             _convexResource != nullptr;
    if (_shapeType == shapeType && !clearConvex)
        return;
    _shapeType = shapeType;
    if (clearConvex)
        _convexResource.reset();
    markChanged();
}

void CollisionShapeComponent::setSize(const glm::vec3& size) {
    requireAttached();
    if (_size == size)
        return;
    _size = size;
    markChanged();
}

void CollisionShapeComponent::setFromTo(const glm::vec3& from,
                                        const glm::vec3& to) {
    requireAttached();
    if (_hasFromTo && _fromPosition == from && _toPosition == to)
        return;
    _hasFromTo = true;
    _fromPosition = from;
    _toPosition = to;
    markChanged();
}

void CollisionShapeComponent::clearFromTo() {
    requireAttached();
    if (!_hasFromTo)
        return;
    _hasFromTo = false;
    _fromPosition = glm::vec3(0.0f);
    _toPosition = glm::vec3(0.0f);
    markChanged();
}

void CollisionShapeComponent::setStaticFriction(float value) {
    requireAttached();
    if (_staticFriction == value)
        return;
    _staticFriction = value;
    markChanged();
}

void CollisionShapeComponent::setDynamicFriction(float value) {
    requireAttached();
    if (_dynamicFriction == value)
        return;
    _dynamicFriction = value;
    markChanged();
}

void CollisionShapeComponent::setRestitution(float value) {
    requireAttached();
    if (_restitution == value)
        return;
    _restitution = value;
    markChanged();
}

void CollisionShapeComponent::setCondim(int value) {
    requireAttached();
    if (_condim == value)
        return;
    _condim = value;
    markChanged();
}

void CollisionShapeComponent::setMargin(float value) {
    requireAttached();
    if (_margin == value)
        return;
    _margin = value;
    markChanged();
}

void CollisionShapeComponent::setSourceGeomIndex(int index) {
    requireAttached();
    if (_sourceGeomIndex == index)
        return;
    _sourceGeomIndex = index;
    markChanged();
}

void CollisionShapeComponent::setConvexResource(
    std::shared_ptr<Physics::ConvexCollisionResource> resource) {
    requireAttached();
    const bool typeChanged =
        resource && _shapeType != CollisionShapeType::ConvexMesh;
    if (_convexResource == resource && !typeChanged)
        return;
    _convexResource = std::move(resource);
    if (_convexResource)
        _shapeType = CollisionShapeType::ConvexMesh;
    markChanged();
}

void CollisionShapeComponent::clearConvexResource() {
    setConvexResource(nullptr);
}

void CollisionShapeComponent::setShapeMetadata(
    CollisionShapeType shapeType, const glm::vec3& size,
    float staticFriction, float dynamicFriction, float restitution, int condim,
    float margin, int sourceGeomIndex) {
    requireAttached();
    const bool changed =
        _shapeType != shapeType || _size != size ||
        _staticFriction != staticFriction ||
        _dynamicFriction != dynamicFriction || _restitution != restitution ||
        _condim != condim || _margin != margin ||
        _sourceGeomIndex != sourceGeomIndex ||
        (shapeType != CollisionShapeType::ConvexMesh && _convexResource);
    if (!changed)
        return;
    _shapeType = shapeType;
    _size = size;
    _staticFriction = staticFriction;
    _dynamicFriction = dynamicFriction;
    _restitution = restitution;
    _condim = condim;
    _margin = margin;
    _sourceGeomIndex = sourceGeomIndex;
    if (shapeType != CollisionShapeType::ConvexMesh)
        _convexResource.reset();
    markChanged();
}

const char* collisionShapeTypeLabel(CollisionShapeType type) {
    switch (type) {
    case CollisionShapeType::Sphere:
        return "Sphere";
    case CollisionShapeType::Capsule:
        return "Capsule";
    case CollisionShapeType::Cylinder:
        return "Cylinder";
    case CollisionShapeType::Box:
        return "Box";
    case CollisionShapeType::ConvexMesh:
        return "Convex Mesh";
    }
    return "Unknown";
}

} // namespace Scene
} // namespace KE

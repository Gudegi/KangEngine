#include "engine/scene/component/collision_shape_component.hpp"

namespace KE {
namespace Scene {

CollisionShapeComponent::CollisionShapeComponent(Prim* owner)
    : ComponentBase(owner, "CollisionShapeComponent") {}

void CollisionShapeComponent::detach() { detachBase(); }

void CollisionShapeComponent::setShapeType(CollisionShapeType shapeType) {
    requireAttached();
    if (_shapeType == shapeType)
        return;
    _shapeType = shapeType;
    markChanged();
}

void CollisionShapeComponent::setSize(const glm::vec3& size) {
    requireAttached();
    if (_size == size)
        return;
    _size = size;
    markChanged();
}

void CollisionShapeComponent::setLocalPosition(const glm::vec3& position) {
    requireAttached();
    if (_localPosition == position)
        return;
    _localPosition = position;
    markChanged();
}

void CollisionShapeComponent::setLocalRotation(const glm::quat& rotation) {
    requireAttached();
    if (_localRotation == rotation)
        return;
    _localRotation = rotation;
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

void CollisionShapeComponent::setShapeMetadata(
    CollisionShapeType shapeType, const glm::vec3& size,
    const glm::vec3& localPosition, const glm::quat& localRotation,
    float staticFriction, float dynamicFriction, float restitution, int condim,
    float margin, int sourceGeomIndex) {
    requireAttached();
    const bool changed =
        _shapeType != shapeType || _size != size ||
        _localPosition != localPosition || _localRotation != localRotation ||
        _staticFriction != staticFriction ||
        _dynamicFriction != dynamicFriction || _restitution != restitution ||
        _condim != condim || _margin != margin ||
        _sourceGeomIndex != sourceGeomIndex;
    if (!changed)
        return;
    _shapeType = shapeType;
    _size = size;
    _localPosition = localPosition;
    _localRotation = localRotation;
    _staticFriction = staticFriction;
    _dynamicFriction = dynamicFriction;
    _restitution = restitution;
    _condim = condim;
    _margin = margin;
    _sourceGeomIndex = sourceGeomIndex;
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
    }
    return "Unknown";
}

} // namespace Scene
} // namespace KE

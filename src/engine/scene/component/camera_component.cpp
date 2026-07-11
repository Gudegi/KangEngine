#include "camera_component.hpp"

#include "engine/scene/native/prim.hpp"

#include <algorithm>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <stdexcept>

namespace KE {
namespace Scene {
namespace {

float sanitizeNear(float value) { return std::max(0.001f, value); }

float sanitizeFar(float nearPlane, float farPlane) {
    return std::max(nearPlane + 0.001f, farPlane);
}

glm::vec3 safeDirection(glm::vec3 direction, glm::vec3 fallback) {
    if (glm::dot(direction, direction) < 1.0e-8f)
        return glm::normalize(fallback);
    return glm::normalize(direction);
}

} // namespace

CameraComponent::CameraComponent(Prim* owner)
    : ComponentBase(owner, "CameraComponent") {}

void CameraComponent::detach() {
    if (!_owner)
        return;
    detachBase();
}

CameraProjectionType CameraComponent::projectionType() const {
    requireAttached();
    return _projectionType;
}

void CameraComponent::setPerspective(float verticalFovDegrees, float nearPlane,
                                     float farPlane) {
    requireAttached();
    _projectionType = CameraProjectionType::Perspective;
    _verticalFovDegrees = std::clamp(verticalFovDegrees, 1.0f, 179.0f);
    _nearPlane = sanitizeNear(nearPlane);
    _farPlane = sanitizeFar(_nearPlane, farPlane);
    markChanged();
}

void CameraComponent::setOrthographic(float verticalSize, float nearPlane,
                                      float farPlane) {
    requireAttached();
    _projectionType = CameraProjectionType::Orthographic;
    _orthographicSize = std::max(0.001f, verticalSize);
    _nearPlane = sanitizeNear(nearPlane);
    _farPlane = sanitizeFar(_nearPlane, farPlane);
    markChanged();
}

float CameraComponent::verticalFovDegrees() const {
    requireAttached();
    return _verticalFovDegrees;
}

float CameraComponent::orthographicSize() const {
    requireAttached();
    return _orthographicSize;
}

float CameraComponent::nearPlane() const {
    requireAttached();
    return _nearPlane;
}

float CameraComponent::farPlane() const {
    requireAttached();
    return _farPlane;
}

glm::vec3 CameraComponent::position() const {
    requireAttached();
    return glm::vec3(_owner->computeWorldMatrix()[3]);
}

glm::vec3 CameraComponent::forward() const {
    requireAttached();
    return safeDirection(glm::mat3(_owner->computeWorldMatrix()) *
                             glm::vec3(0.0f, 0.0f, -1.0f),
                         glm::vec3(0.0f, 0.0f, -1.0f));
}

glm::vec3 CameraComponent::up() const {
    requireAttached();
    return safeDirection(glm::mat3(_owner->computeWorldMatrix()) *
                             glm::vec3(0.0f, 1.0f, 0.0f),
                         glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 CameraComponent::viewMatrix() const {
    requireAttached();
    return glm::inverse(_owner->computeWorldMatrix());
}

glm::mat4 CameraComponent::projectionMatrix(float aspect) const {
    requireAttached();
    const float safeAspect = std::max(0.001f, aspect);
    if (_projectionType == CameraProjectionType::Orthographic) {
        const float halfHeight = _orthographicSize * 0.5f;
        const float halfWidth = halfHeight * safeAspect;
        return glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight,
                          _nearPlane, _farPlane);
    }
    return glm::perspective(glm::radians(_verticalFovDegrees), safeAspect,
                            _nearPlane, _farPlane);
}

glm::mat4 CameraComponent::viewProjectionMatrix(float aspect) const {
    requireAttached();
    return projectionMatrix(aspect) * viewMatrix();
}

} // namespace Scene
} // namespace KE

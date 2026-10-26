#include "light_component.hpp"

#include "engine/scene/native/prim.hpp"

#include <algorithm>
#include <cmath>
#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <stdexcept>

namespace KE {
namespace Scene {

namespace {
glm::vec3 safeDirection(glm::vec3 direction, glm::vec3 fallback) {
    if (glm::length(direction) < 1e-4f)
        return glm::normalize(fallback);
    return glm::normalize(direction);
}

void validateEmission(const LightEmission& light) {
    if (!std::isfinite(light.intensity) || light.intensity < 0 ||
        !std::isfinite(light.exposure) || !std::isfinite(std::exp2(light.exposure) * light.intensity) ||
        !std::isfinite(light.colorTemperature) || light.colorTemperature <= 0 ||
        !std::isfinite(light.diffuse) || light.diffuse < 0 ||
        !std::isfinite(light.specular) || light.specular < 0 ||
        !std::isfinite(light.color.r) || !std::isfinite(light.color.g) ||
        !std::isfinite(light.color.b) || glm::any(glm::lessThan(light.color, glm::vec3(0))))
        throw std::invalid_argument("Light emission must be finite and nonnegative; temperature must be positive");
}
} // namespace

LightComponent::LightComponent(Prim* owner)
    : ComponentBase(owner, "LightComponent") {}

void LightComponent::detach() {
    if (!_owner)
        return;
    detachBase();
}

LightType LightComponent::type() const {
    requireAttached();
    return _type;
}

void LightComponent::setDirectionalLight(const DirectionalLight& light) {
    requireAttached();
    _type = LightType::Directional;
    _direction = safeDirection(light.direction, glm::vec3(0.0f, 0.0f, -1.0f));
    _color = light.color;
    _intensity = std::max(0.0f, light.intensity);
    _ambient = light.ambient;
    markChanged();
}

DirectionalLight LightComponent::directionalLight() const {
    requireAttached();
    DirectionalLight light;
    const glm::mat3 worldRotation(_owner->computeWorldMatrix());
    light.direction =
        safeDirection(worldRotation * _direction, light.direction);
    light.color = _color;
    light.intensity = std::max(0.0f, _intensity);
    light.ambient = _ambient;
    return light;
}

void LightComponent::setPointLight(const PointLight& light) {
    requireAttached();
    _type = LightType::Point;
    _owner->setLocalTranslation(light.position);
    _color = light.color;
    _intensity = std::max(0.0f, light.intensity);
    _range = std::max(0.0f, light.range);
    markChanged();
}

PointLight LightComponent::pointLight() const {
    requireAttached();
    PointLight light;
    const glm::mat4 world = _owner->computeWorldMatrix();
    light.position = glm::vec3(world[3]);
    light.color = _color;
    light.intensity = std::max(0.0f, _intensity);
    light.range = std::max(0.0f, _range);
    return light;
}

void LightComponent::setSpotLight(const SpotLight& light) {
    requireAttached();
    _type = LightType::Spot;
    _owner->setLocalTranslation(light.position);
    _direction = safeDirection(light.direction, glm::vec3(0.0f, 0.0f, -1.0f));
    _color = light.color;
    _intensity = std::max(0.0f, light.intensity);
    _range = std::max(0.0f, light.range);
    _innerConeAngle = std::max(0.0f, light.innerConeAngle);
    _outerConeAngle = std::max(_innerConeAngle, light.outerConeAngle);
    markChanged();
}

SpotLight LightComponent::spotLight() const {
    requireAttached();
    SpotLight light;
    const glm::mat4 world = _owner->computeWorldMatrix();
    light.position = glm::vec3(world[3]);
    light.direction =
        safeDirection(glm::mat3(world) * _direction, light.direction);
    light.color = _color;
    light.intensity = std::max(0.0f, _intensity);
    light.range = std::max(0.0f, _range);
    light.innerConeAngle = std::max(0.0f, _innerConeAngle);
    light.outerConeAngle = std::max(light.innerConeAngle, _outerConeAngle);
    return light;
}

void LightComponent::setRectLight(const RectLight& light) {
    requireAttached();
    validateEmission(light);
    if (!std::isfinite(light.width) || !std::isfinite(light.height) ||
        light.width <= 0 || light.height <= 0)
        throw std::invalid_argument("RectLight dimensions must be finite and positive");
    _type = LightType::Rect;
    _rect = light;
    markChanged();
}

RectLight LightComponent::rectLight() const {
    requireAttached();
    return _rect;
}

void LightComponent::setDomeLight(const DomeLight& light) {
    requireAttached();
    validateEmission(light);
    _type = LightType::Dome;
    _dome = light;
    markChanged();
}

DomeLight LightComponent::domeLight() const {
    requireAttached();
    return _dome;
}

} // namespace Scene
} // namespace KE

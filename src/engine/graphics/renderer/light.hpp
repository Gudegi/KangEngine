#pragma once

#include "engine/scene/types/light.hpp"

namespace KE {

inline constexpr int MaxPointLights = 4;
inline constexpr int MaxSpotLights = 2;
inline constexpr int MaxRectLights = 16;

// Evaluated render data; authored dimensions/emission remain in LightComponent.
struct RectLightInstance {
    Scene::RectLight light;
    glm::mat4 worldTransform{1.0f};
};

using DirectionalLight = Scene::DirectionalLight;
using PointLight = Scene::PointLight;
using SpotLight = Scene::SpotLight;

// Evaluate prim-local authoring for rendering and world-space overlays.
// The matrix already includes the light prim's translation; do not add position twice.
inline glm::vec3 worldLightDirection(const glm::mat4& world, glm::vec3 direction) {
    const glm::vec3 transformed = glm::mat3(world) * direction;
    const float length = glm::length(transformed);
    return length > 1e-4f ? transformed / length : glm::vec3(0, 0, -1);
}

inline DirectionalLight evaluateWorldLight(DirectionalLight light, const glm::mat4& world) {
    light.direction = worldLightDirection(world, light.direction);
    return light;
}

inline PointLight evaluateWorldLight(PointLight light, const glm::mat4& world) {
    light.position = glm::vec3(world[3]);
    return light;
}

inline SpotLight evaluateWorldLight(SpotLight light, const glm::mat4& world) {
    light.position = glm::vec3(world[3]);
    light.direction = worldLightDirection(world, light.direction);
    return light;
}

} // namespace KE

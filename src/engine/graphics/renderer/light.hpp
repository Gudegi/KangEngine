#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace KE {

inline constexpr int MaxPointLights = 4;
inline constexpr int MaxSpotLights = 2;

/// Directional light — infinite distance, no position.
///
/// "direction" is the world-space vector pointing toward the light source
/// (i.e. the "L" vector in Phong: dot(N, L) > 0 means lit).
/// Example: (0, 0, 1) -> light comes from -Z (shines in +Z direction).
struct DirectionalLight {
    glm::vec3 direction = glm::normalize(glm::vec3(0.2f, 0.5f, 1.0f));
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 0.75f;
    glm::vec3 ambient = {0.15f, 0.15f, 0.15f};
};

/// Point light — finite local light with distance falloff.
///
/// "range" is the artist-facing influence radius. Forward PBR consumes up to
/// MaxPointLights point lights without shadows.
struct PointLight {
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float range = 10.0f;
};

/// Spot light — finite cone light with distance and angular falloff.
///
/// "direction" points from the light toward the target. Cone angles are stored
/// in radians; innerConeAngle should be <= outerConeAngle.
struct SpotLight {
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 direction = {0.0f, 0.0f, -1.0f};
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float range = 10.0f;
    float innerConeAngle = glm::radians(20.0f);
    float outerConeAngle = glm::radians(30.0f);
};

} // namespace KE

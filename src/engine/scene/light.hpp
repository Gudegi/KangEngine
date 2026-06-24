#ifndef _LIGHT_HPP_
#define _LIGHT_HPP_

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace KE {
namespace Scene {

enum class LightType {
    Directional,
    Point,
    Spot,
};

/// Scene-level directional light — infinite distance, no position.
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

/// Scene-level point light — finite local light with distance falloff.
struct PointLight {
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float range = 10.0f;
};

/// Scene-level spot light — finite cone light with distance and angular
/// falloff.
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

} // namespace Scene
} // namespace KE

#endif
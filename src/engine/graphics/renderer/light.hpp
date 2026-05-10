#pragma once
#include <glm/glm.hpp>

namespace KE {

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

} // namespace KE

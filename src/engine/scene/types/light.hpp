#ifndef _SCENE_TYPES_LIGHT_HPP_
#define _SCENE_TYPES_LIGHT_HPP_

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <string>

namespace KE {
namespace Scene {

enum class LightType {
    Directional,
    Point,
    Spot,
    Rect,
    Dome,
};

/// Scene-level directional light — infinite distance, no position.
///
/// "direction" is the world-space vector pointing toward the light source
/// (i.e. the "L" vector in Phong: dot(N, L) > 0 means lit).
/// Example: (0, 0, 1) -> light comes from -Z (shines in +Z direction).
struct DirectionalLight {
    glm::vec3 direction = glm::normalize(glm::vec3(0.2f, 0.5f, 1.0f));
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 2.0f;
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

// Authored UsdLux-compatible emission. Exposure is in stops; intensity is
// luminance before exposure and optional world-area normalization.
struct LightEmission {
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float exposure = 0.0f;
    bool normalize = false;
    bool enableColorTemperature = false;
    float colorTemperature = 6500.0f;
    float diffuse = 1.0f;
    float specular = 1.0f;
};

// Centered in the prim's local XY plane, emitting along local -Z. Transform
// belongs to the prim, so authored dimensions survive parent scale/rotation.
struct RectLight : LightEmission {
    float width = 1.0f;
    float height = 1.0f;
    std::string texturePath;
};

// Uniform infinite environment. Textured dome lighting is not yet evaluated.
struct DomeLight : LightEmission {
    std::string texturePath;
    std::string textureFormat = "automatic";
};

} // namespace Scene
} // namespace KE

#endif

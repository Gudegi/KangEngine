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

} // namespace KE

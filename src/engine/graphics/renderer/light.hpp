#pragma once

#include "engine/scene/types/light.hpp"

namespace KE {

inline constexpr int MaxPointLights = 4;
inline constexpr int MaxSpotLights = 2;

using DirectionalLight = Scene::DirectionalLight;
using PointLight = Scene::PointLight;
using SpotLight = Scene::SpotLight;

} // namespace KE

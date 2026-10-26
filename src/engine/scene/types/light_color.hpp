// Copyright 2016 Pixar. Adapted from OpenUSD usdLux/blackbody.cpp to GLM.
#pragma once

#include "engine/scene/types/light.hpp"
#include <algorithm>
#include <cmath>

namespace KE::Scene {

inline glm::vec3 blackbodyColor(float kelvin) {
    // Linear Rec.709 RGB, 1000–10000 K in 500 K steps. Repeated end knots
    // keep Catmull-Rom boundary behavior consistent with UsdLux.
    static const glm::vec3 colors[] = {
        {1, .027490f, 0}, {1, .027490f, 0}, {1, .149664f, 0},
        {1, .256644f, .008095f}, {1, .372033f, .067450f},
        {1, .476725f, .153601f}, {1, .570376f, .259196f},
        {1, .653480f, .377155f}, {1, .726878f, .501606f},
        {1, .791543f, .628050f}, {1, .848462f, .753228f},
        {1, .898581f, .874905f}, {1, .942771f, .991642f},
        {.906947f, .890456f, 1}, {.828247f, .841838f, 1},
        {.765791f, .801896f, 1}, {.715255f, .768579f, 1},
        {.673683f, .740423f, 1}, {.638992f, .716359f, 1},
        {.609681f, .695588f, 1}, {.609681f, .695588f, 1},
        {.609681f, .695588f, 1},
    };
    const float x = std::clamp((kelvin - 1000.0f) / 500.0f, 0.0f, 18.0f);
    const int segment = static_cast<int>(x);
    const float t = x - segment;
    const auto a = colors[segment], b = colors[segment + 1];
    const auto c = colors[segment + 2], d = colors[segment + 3];
    glm::vec3 rgb = 0.5f * ((-a + 3.0f*b - 3.0f*c + d)*t*t*t +
                            (2.0f*a - 5.0f*b + 4.0f*c - d)*t*t +
                            (-a + c)*t + 2.0f*b);
    rgb /= glm::dot(rgb, glm::vec3(.2126f, .7152f, .0722f));
    return glm::max(rgb, glm::vec3(0));
}

inline glm::vec3 emittedRadiance(const LightEmission& light) {
    const glm::vec3 temperature = light.enableColorTemperature
        ? blackbodyColor(light.colorTemperature) : glm::vec3(1);
    return light.color * temperature * (light.intensity * std::exp2(light.exposure));
}

} // namespace KE::Scene

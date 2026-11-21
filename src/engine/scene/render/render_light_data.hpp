#pragma once

// Scene light values and world transforms consumed by render backends.

#include "engine/scene/types/light.hpp"

#include <string>
#include <variant>
#include <vector>

namespace KE {
namespace Scene {

struct RenderLightSource {
    std::string path;
    std::variant<DirectionalLight, PointLight, SpotLight, RectLight, DomeLight>
        light;
    glm::mat4 world{1.0f};
};

struct RenderLightData {
    std::vector<RenderLightSource> lights;
    // Hidden lights still make the scene the owner of lighting state.
    bool authored = false;
};

} // namespace Scene
} // namespace KE

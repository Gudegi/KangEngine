#pragma once

// Scene light inputs and membership collection for render backends.

#include "engine/scene/types/light.hpp"

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace KE {
namespace Scene {

class Prim;
class TransformComponent;

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

// Caches membership; visibility and world poses are evaluated on each
// collection.
class RenderLightCollector {
  public:
    RenderLightData collect(Prim* root);

  private:
    std::weak_ptr<TransformComponent> _root;
    uint64_t _structureVersion = 0;
    std::vector<Prim*> _prims;
};

} // namespace Scene
} // namespace KE

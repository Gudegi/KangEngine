#ifndef __RENDERER_HPP__
#define __RENDERER_HPP__

#include "scene/scene_backend.hpp"
#include "backend/base/graphics_device.hpp"
#include <glm/mat4x4.hpp>

namespace KE {
class Renderer {
  public:
    Backend::GraphicsDevice* _graphicsDevice;
    Scene::SceneBackend* _scene;

    virtual void render(const glm::mat4& view, const glm::mat4& proj) = 0;
    virtual ~Renderer() = default;
};

} // namespace KE
#endif
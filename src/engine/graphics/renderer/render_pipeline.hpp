#ifndef _RENDER_PIPELINE_HPP_
#define _RENDER_PIPELINE_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/scene/scene_backend.hpp"

#include <glm/mat4x4.hpp>

namespace KE {

class RenderPipeline {
  public:
    Backend::GraphicsDevice* _graphicsDevice = nullptr;
    Scene::SceneBackend* _scene = nullptr;

    virtual void render(const glm::mat4& view, const glm::mat4& proj,
                        Backend::RenderTarget* target) = 0;
    virtual ~RenderPipeline() = default;
};

} // namespace KE

#endif

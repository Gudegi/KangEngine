#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"
#include <memory>

namespace KE {

// current support post-processing : gamma correction

class PostProcessor {
  public:
    void init(Backend::GraphicsDevice* device, int width, int height);
    void process(Backend::Texture* src, float gamma);
    Backend::Texture* getResult();
    void blitToScreen(int width, int height);
    void resize(int width, int height);

  private:
    Backend::GraphicsDevice* _device = nullptr;
    std::unique_ptr<Backend::Shader> _shader;
    std::unique_ptr<Backend::VertexArray> _quadVAO;
    std::unique_ptr<Backend::Buffer> _posVBO;
    std::unique_ptr<Backend::Buffer> _uvVBO;
    std::unique_ptr<Backend::Buffer> _ibo;
    std::unique_ptr<Backend::Framebuffer> _outputFBO;
    int _width = 0, _height = 0;
};

} // namespace KE

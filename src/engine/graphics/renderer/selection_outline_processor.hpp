#ifndef _SELECTION_OUTLINE_PROCESSOR_HPP_
#define _SELECTION_OUTLINE_PROCESSOR_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include <glm/vec4.hpp>
#include <memory>

namespace KE {

struct SelectionOutlineConfig {
    bool enabled = true;
    glm::vec4 color = glm::vec4(1.0f, 0.58f, 0.0f, 1.0f);
    float radius = 2.0f;
};

class SelectionOutlineProcessor {
  public:
    void init(Backend::GraphicsDevice* device, int width, int height);
    void process(Backend::Texture* sceneColor, Backend::Texture* selectionMask);
    Backend::Texture* getResult();
    Backend::Framebuffer* getOutputFramebuffer();
    void blitToScreen(int width, int height);
    void resize(int width, int height);
    SelectionOutlineConfig& config() { return _config; }
    const SelectionOutlineConfig& config() const { return _config; }
    void setOutlineColor(const glm::vec4& color) { _config.color = color; }
    void setOutlineRadius(float radius) { _config.radius = radius; }

  private:
    Backend::GraphicsDevice* _device = nullptr;
    std::unique_ptr<Backend::Shader> _shader;
    std::unique_ptr<Backend::VertexArray> _quadVAO;
    std::unique_ptr<Backend::Buffer> _posVBO;
    std::unique_ptr<Backend::Buffer> _uvVBO;
    std::unique_ptr<Backend::Buffer> _ibo;
    std::unique_ptr<Backend::Framebuffer> _outputFBO;
    SelectionOutlineConfig _config;
    int _width = 0;
    int _height = 0;
};

} // namespace KE

#endif

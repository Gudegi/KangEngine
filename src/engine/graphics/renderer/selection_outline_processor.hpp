#ifndef _SELECTION_OUTLINE_PROCESSOR_HPP_
#define _SELECTION_OUTLINE_PROCESSOR_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include <array>
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
    void renderOutlineCompositePass(Backend::Texture* sceneColor,
                                    Backend::Texture* selectionMask);
    Backend::Texture* getResult();
    Backend::Framebuffer* getOutputFramebuffer();
    void blitToScreen(int width, int height);
    void resize(int width, int height);
    SelectionOutlineConfig& config() { return _config; }
    const SelectionOutlineConfig& config() const { return _config; }
    void setOutlineColor(const glm::vec4& color) { _config.color = color; }
    void setOutlineRadius(float radius) { _config.radius = radius; }

  private:
    struct alignas(16) OutlineParams {
        glm::vec4 texelSizeAndRadius{0.0f};
        glm::vec4 color{1.0f};
    };

    Backend::GraphicsDevice* _device = nullptr;
    std::unique_ptr<Backend::Framebuffer> _outputFBO;
    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4> _groupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _pipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _pipeline;
    std::unique_ptr<Backend::Sampler> _sampler;
    std::unique_ptr<Backend::Buffer> _paramsBuffer;
    std::unique_ptr<Backend::TextureView> _outputView;
    std::unique_ptr<Backend::RenderTarget> _outputTarget;
    std::unique_ptr<Backend::TextureView> _sceneView;
    std::unique_ptr<Backend::TextureView> _maskView;
    std::unique_ptr<Backend::BindGroup> _passBindGroup;
    Backend::Texture* _boundScene = nullptr;
    Backend::Texture* _boundMask = nullptr;
    SelectionOutlineConfig _config;
    int _width = 0;
    int _height = 0;

    void rebuildOutputTarget();
    void ensurePassBindings(Backend::Texture* sceneColor,
                            Backend::Texture* selectionMask);
};

} // namespace KE

#endif

#pragma once

#include "engine/graphics/renderer/render_pass_base.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"

#include <array>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <memory>
#include <vector>

namespace KE {

class MeshInstancer;
class ShaderLibrary;

// On-demand surface picking; source meshes and normal instance buffers stay
// intact.
class PickingPass : public RenderPassBase {
  public:
    PickingPass() : RenderPassBase("PickingPass") {}
    RayPickResult pick(const std::vector<MeshInstancer*>& instancers,
                       ShaderLibrary& shaders, const glm::mat4& view,
                       const glm::mat4& projection, glm::vec2 ndc, int width,
                       int height);

  private:
    void initializeResources();
    Backend::GraphicsPipeline* pipeline(ShaderLibrary& shaders, bool skinned,
                                        bool textured, bool doubleSided);

    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4> _layouts;
    std::unique_ptr<Backend::BindGroupLayout> _emptyLayout;
    std::array<std::unique_ptr<Backend::PipelineLayout>, 4> _pipelineLayouts;
    std::array<std::unique_ptr<Backend::GraphicsPipeline>, 8> _pipelines;
    std::unique_ptr<Backend::Buffer> _camera;
    std::unique_ptr<Backend::BindGroup> _frame;
    std::unique_ptr<Backend::Sampler> _sampler;
    std::unique_ptr<Backend::Texture> _color;
    std::unique_ptr<Backend::Texture> _depth;
    std::unique_ptr<Backend::TextureView> _colorView;
    std::unique_ptr<Backend::TextureView> _depthView;
    std::unique_ptr<Backend::RenderTarget> _target;
};

} // namespace KE

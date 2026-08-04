#pragma once

#include "engine/graphics/renderer/render_pass_base.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace KE {

class SkyboxPass : public RenderPassBase {
  public:
    SkyboxPass() : RenderPassBase("SkyboxPass") {}

    void initializeResources(Backend::BindGroupLayout* frameLayout);
    void setTexture(const std::string& path, UpAxis upAxis);
    void setTexture(const std::vector<std::string>& paths, UpAxis upAxis);
    bool ready() const { return _pipeline && _textureBindGroup; }
    void record(Backend::RenderPassEncoder& pass,
                Backend::BindGroup* frameBindGroup) const;

  private:
    void rebuildBinding(UpAxis upAxis);

    std::unique_ptr<Backend::BindGroupLayout> _passGroupLayout;
    std::unique_ptr<Backend::BindGroupLayout> _textureGroupLayout;
    std::unique_ptr<Backend::PipelineLayout> _pipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _pipeline;
    std::unique_ptr<Backend::Buffer> _vertexBuffer;
    std::unique_ptr<Backend::Buffer> _indexBuffer;
    std::unique_ptr<Backend::Buffer> _paramsBuffer;
    std::unique_ptr<Backend::BindGroup> _paramsBindGroup;
    std::unique_ptr<Backend::Texture> _texture;
    std::unique_ptr<Backend::TextureView> _textureView;
    std::unique_ptr<Backend::Sampler> _sampler;
    std::unique_ptr<Backend::BindGroup> _textureBindGroup;
};

} // namespace KE

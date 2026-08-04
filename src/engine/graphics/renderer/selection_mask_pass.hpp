#pragma once

#include "engine/graphics/renderer/render_pass_base.hpp"

#include <array>
#include <memory>

namespace KE {

class MeshInstancer;

class SelectionMaskPass : public RenderPassBase {
  public:
    SelectionMaskPass() : RenderPassBase("SelectionMaskPass") {}

    struct PreparedDraw {
        MeshInstancer* instancer = nullptr;
        Backend::GraphicsPipeline* pipeline = nullptr;
        std::unique_ptr<Backend::TextureView> alphaView;
        std::unique_ptr<Backend::BindGroup> alphaBindGroup;
        std::unique_ptr<Backend::BindGroup> skinBindGroup;
        bool alphaMask = false;
        bool skinned = false;
        int instanceIndex = -1;
    };

    void initializeResources(Backend::Buffer* cameraBuffer);
    PreparedDraw prepare(MeshInstancer* instancer, int instanceIndex,
                         Backend::Framebuffer* target);
    Backend::RenderTarget* target() const { return _outputTarget.get(); }
    void record(Backend::RenderPassEncoder& pass,
                const PreparedDraw& draw) const;

  private:
    static size_t index(bool skinned, bool alphaMask, bool doubleSided);
    void ensureTarget(Backend::Framebuffer* target);

    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4> _groupLayouts;
    std::unique_ptr<Backend::BindGroupLayout> _alphaGroupLayout;
    std::unique_ptr<Backend::BindGroupLayout> _skinGroupLayout;
    std::array<std::unique_ptr<Backend::PipelineLayout>, 4> _pipelineLayouts;
    std::array<std::unique_ptr<Backend::GraphicsPipeline>, 8> _pipelines;
    std::unique_ptr<Backend::BindGroup> _frameBindGroup;
    std::unique_ptr<Backend::Sampler> _alphaSampler;
    std::unique_ptr<Backend::TextureView> _outputView;
    std::unique_ptr<Backend::RenderTarget> _outputTarget;
    uintptr_t _outputHandle = 0;
    int _outputWidth = 0;
    int _outputHeight = 0;
};

} // namespace KE

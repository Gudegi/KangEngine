#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/renderer/render_pass_base.hpp"

#include <array>
#include <memory>
#include <vector>

namespace KE {

class MeshInstancer;

class ShadowPass : public RenderPassBase {
  public:
    static constexpr size_t MaxCascades = 4;
    ShadowPass() : RenderPassBase("ShadowPass") {}
    struct PreparedDraw {
        MeshInstancer* instancer = nullptr;
        Backend::GraphicsPipeline* pipeline = nullptr;
        Backend::BindGroup* skinBindGroup = nullptr;
        Backend::BindGroup* alphaBindGroup = nullptr;
        bool alphaMask = false;
        bool skinned = false;
    };
    struct PreparedDraws {
        std::vector<PreparedDraw> draws;
        std::vector<std::unique_ptr<Backend::TextureView>> ownedViews;
        std::vector<std::unique_ptr<Backend::BindGroup>> ownedBindGroups;
    };

    void createPipeline(bool skinned, bool alphaMask, bool doubleSided,
                        const Backend::GraphicsPipelineDesc& desc);
    Backend::GraphicsPipeline* pipelineFor(bool skinned, bool alphaMask,
                                           bool doubleSided) const;
    void initializeResources(Backend::Buffer* shadowBuffer);
    PreparedDraws prepare(const std::vector<MeshInstancer*>& casters);
    void record(Backend::RenderPassEncoder& pass,
                const PreparedDraws& prepared) const;
    void initializeSampling();
    void
    rebuildSampling(const std::array<Backend::Texture*, MaxCascades>& textures,
                    bool debugCascadeTint);
    Backend::BindGroupLayout* samplingLayout() const {
        return _samplingLayout.get();
    }
    Backend::BindGroup* samplingBindGroup() const {
        return _samplingBindGroup.get();
    }
    void initializeDepthTargets(
        Backend::Framebuffer* single,
        const std::array<Backend::Framebuffer*, MaxCascades>& cascades);
    Backend::RenderTarget* singleDepthTarget() const {
        return _singleDepthTarget.get();
    }
    Backend::RenderTarget* cascadeDepthTarget(size_t index) const {
        return index < _cascadeDepthTargets.size()
                   ? _cascadeDepthTargets[index].get()
                   : nullptr;
    }

  private:
    static size_t index(bool skinned, bool alphaMask, bool doubleSided);

    std::array<std::unique_ptr<Backend::GraphicsPipeline>, 8> _pipelines;
    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4> _groupLayouts;
    std::unique_ptr<Backend::BindGroupLayout> _alphaGroupLayout;
    std::unique_ptr<Backend::BindGroupLayout> _skinGroupLayout;
    std::array<std::unique_ptr<Backend::PipelineLayout>, 4> _pipelineLayouts;
    std::unique_ptr<Backend::BindGroup> _frameBindGroup;
    std::unique_ptr<Backend::Sampler> _alphaSampler;
    std::unique_ptr<Backend::BindGroupLayout> _samplingLayout;
    std::unique_ptr<Backend::Sampler> _samplingSampler;
    std::unique_ptr<Backend::Buffer> _samplingParamsBuffer;
    std::array<std::unique_ptr<Backend::TextureView>, MaxCascades>
        _samplingViews;
    std::unique_ptr<Backend::BindGroup> _samplingBindGroup;
    std::array<uintptr_t, MaxCascades> _samplingHandles{};
    std::unique_ptr<Backend::TextureView> _singleDepthView;
    std::unique_ptr<Backend::RenderTarget> _singleDepthTarget;
    std::array<std::unique_ptr<Backend::TextureView>, MaxCascades>
        _cascadeDepthViews;
    std::array<std::unique_ptr<Backend::RenderTarget>, MaxCascades>
        _cascadeDepthTargets;
};

} // namespace KE

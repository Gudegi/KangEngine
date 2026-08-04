#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"
#include <array>
#include <memory>

namespace KE {

// Flow:
// scene HDR -> optional bright extract -> ping-pong blur -> bloom composite
//           -> tone mapping + gamma -> output LDR framebuffer -> screen

enum class ToneMapMode {
    None = 0,
    Reinhard = 1,
    Exponential = 2,
    AcesNarkowicz = 3,
    AcesFitted = 4,
};

// Optional bloom pass settings. Bloom is skipped entirely when disabled.
struct BloomConfig {
    // Enables the bloom extract/blur/composite chain.
    bool enabled = false;
    // Extract only HDR pixels brighter than this value.
    float threshold = 1.0f;
    // Additive strength when blurred bloom is composited back into the scene.
    float intensity = 0.08f;
    // Ping-pong blur pass count. Higher values make bloom wider and softer.
    int iterations = 6;
    // Bloom buffer size divisor. 2 means half-resolution bloom buffers.
    int downsample = 2;
};

class PostProcessor {
  public:
    void init(Backend::GraphicsDevice* device, int width, int height);
    void process(Backend::Texture* src, float gamma, ToneMapMode toneMapMode,
                 float tonemapExposure, const BloomConfig& bloom);
    Backend::Texture* getResult();
    Backend::Framebuffer* getOutputFramebuffer();
    void blitToScreen(int width, int height);
    void resize(int width, int height);

  private:
    struct alignas(16) ToneMapParams {
        // x: gamma, y: tone-map mode, z: exposure
        glm::vec4 values{1.0f, 0.0f, 1.0f, 0.0f};
    };
    struct alignas(16) BrightExtractParams {
        glm::vec4 values{1.0f, 0.0f, 0.0f, 0.0f};
    };
    struct alignas(16) BlurParams {
        glm::vec4 values{0.0f};
    };
    struct alignas(16) BloomCompositeParams {
        glm::vec4 values{0.0f};
    };

    Backend::GraphicsDevice* _device = nullptr;

    // Final LDR output after bloom, tone mapping, and gamma correction.
    std::unique_ptr<Backend::Framebuffer> _outputFBO;

    // RHI resources for the final tone-map/gamma fullscreen pass. Legacy
    // bloom stages remain below until they are migrated independently.
    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4>
        _toneMapGroupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _toneMapPipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _toneMapPipeline;
    std::unique_ptr<Backend::Sampler> _toneMapSampler;
    std::unique_ptr<Backend::Buffer> _toneMapParamsBuffer;
    std::unique_ptr<Backend::TextureView> _toneMapOutputView;
    std::unique_ptr<Backend::RenderTarget> _toneMapOutputTarget;
    std::unique_ptr<Backend::TextureView> _toneMapSourceView;
    std::unique_ptr<Backend::BindGroup> _toneMapBindGroup;
    Backend::Texture* _boundToneMapSource = nullptr;

    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4>
        _brightExtractGroupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _brightExtractPipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _brightExtractPipeline;
    std::unique_ptr<Backend::Sampler> _brightExtractSampler;
    std::unique_ptr<Backend::Buffer> _brightExtractParamsBuffer;
    std::unique_ptr<Backend::TextureView> _brightExtractOutputView;
    std::unique_ptr<Backend::RenderTarget> _brightExtractOutputTarget;
    std::unique_ptr<Backend::TextureView> _brightExtractSourceView;
    std::unique_ptr<Backend::BindGroup> _brightExtractBindGroup;
    Backend::Texture* _boundBrightExtractSource = nullptr;

    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4> _blurGroupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _blurPipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _blurPipeline;
    std::unique_ptr<Backend::Sampler> _blurSampler;
    std::unique_ptr<Backend::Buffer> _blurParamsBuffer;
    std::array<std::unique_ptr<Backend::TextureView>, 2> _blurOutputViews;
    std::array<std::unique_ptr<Backend::RenderTarget>, 2> _blurOutputTargets;
    std::array<std::unique_ptr<Backend::TextureView>, 2> _blurSourceViews;
    std::array<std::unique_ptr<Backend::BindGroup>, 2> _blurBindGroups;
    std::array<Backend::Texture*, 2> _boundBlurSources{nullptr, nullptr};

    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4>
        _bloomCompositeGroupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _bloomCompositePipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _bloomCompositePipeline;
    std::unique_ptr<Backend::Sampler> _bloomCompositeSampler;
    std::unique_ptr<Backend::Buffer> _bloomCompositeParamsBuffer;
    std::unique_ptr<Backend::TextureView> _bloomCompositeOutputView;
    std::unique_ptr<Backend::RenderTarget> _bloomCompositeOutputTarget;
    std::unique_ptr<Backend::TextureView> _bloomCompositeSceneView;
    std::unique_ptr<Backend::TextureView> _bloomCompositeBloomView;
    std::unique_ptr<Backend::BindGroup> _bloomCompositeBindGroup;
    Backend::Texture* _boundCompositeScene = nullptr;
    Backend::Texture* _boundCompositeBloom = nullptr;

    // Half/quarter/etc resolution RGBA16F buffers used alternately for blur.
    std::array<std::unique_ptr<Backend::Framebuffer>, 2> _bloomPingPongFBO;

    // Full-resolution HDR scene + blurred bloom composite before tone mapping.
    std::unique_ptr<Backend::Framebuffer> _bloomCompositeFBO;

    // Main post-process size and cached bloom buffer size.
    int _width = 0, _height = 0;
    int _bloomWidth = 0, _bloomHeight = 0, _bloomDownsample = 0;

    // Lazily creates/resizes bloom buffers only when bloom is enabled.
    void ensureBloomBuffers(const BloomConfig& bloom);
    void initToneMapRhi();
    void rebuildToneMapOutputTarget();
    void ensureToneMapBindings(Backend::Texture* source);
    void initBrightExtractRhi();
    void rebuildBrightExtractOutputTarget();
    void ensureBrightExtractBindings(Backend::Texture* source);
    void initBlurRhi();
    void rebuildBlurOutputTargets();
    Backend::BindGroup* ensureBlurBinding(Backend::Texture* source);
    void initBloomCompositeRhi();
    void rebuildBloomCompositeOutputTarget();
    void ensureBloomCompositeBindings(Backend::Texture* scene,
                                      Backend::Texture* bloom);

    // Writes thresholded HDR highlights into the first bloom buffer.
    void renderBrightExtractPass(Backend::Texture* src,
                                 Backend::Framebuffer* target, float threshold);

    // Blurs the bright texture through ping-pong FBOs and returns final blur.
    Backend::Texture* renderBloomBlurPass(Backend::Texture* src,
                                          const BloomConfig& bloom);

    // Adds blurred bloom to the original HDR scene.
    void renderBloomCompositePass(Backend::Texture* scene,
                                  Backend::Texture* bloom,
                                  Backend::Framebuffer* target,
                                  float intensity);

    // Converts final HDR source into the LDR output framebuffer.
    void renderToneMapPass(Backend::Texture* src, float gamma,
                           ToneMapMode toneMapMode, float tonemapExposure);
};

} // namespace KE

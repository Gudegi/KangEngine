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
    Backend::GraphicsDevice* _device = nullptr;

    // Fullscreen shaders for each post-processing stage.
    std::unique_ptr<Backend::Shader> _toneMapShader;
    std::unique_ptr<Backend::Shader> _brightExtractShader;
    std::unique_ptr<Backend::Shader> _blurShader;
    std::unique_ptr<Backend::Shader> _bloomCompositeShader;

    // Shared fullscreen quad geometry(to attach texture) used by all
    // post-processing passes.
    std::unique_ptr<Backend::VertexArray> _quadVAO;
    std::unique_ptr<Backend::Buffer> _posVBO;
    std::unique_ptr<Backend::Buffer> _uvVBO;
    std::unique_ptr<Backend::Buffer> _ibo;

    // Final LDR output after bloom, tone mapping, and gamma correction.
    std::unique_ptr<Backend::Framebuffer> _outputFBO;

    // Half/quarter/etc resolution RGBA16F buffers used alternately for blur.
    std::array<std::unique_ptr<Backend::Framebuffer>, 2> _bloomPingPongFBO;

    // Full-resolution HDR scene + blurred bloom composite before tone mapping.
    std::unique_ptr<Backend::Framebuffer> _bloomCompositeFBO;

    // Main post-process size and cached bloom buffer size.
    int _width = 0, _height = 0;
    int _bloomWidth = 0, _bloomHeight = 0, _bloomDownsample = 0;

    // Lazily creates/resizes bloom buffers only when bloom is enabled.
    void ensureBloomBuffers(const BloomConfig& bloom);
    void drawFullscreen();

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

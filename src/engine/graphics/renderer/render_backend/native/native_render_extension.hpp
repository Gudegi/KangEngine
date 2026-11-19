#pragma once

// Native-only access and GPU resources for custom draws and offscreen targets.

#include "engine/graphics/renderer/renderer.hpp"

namespace KE {

class NativeRenderExtension {
  public:
    // Borrowed device, Rasterizer, and settings outlive this extension.
    NativeRenderExtension(Backend::GraphicsDevice& device,
                          Rasterizer& rasterizer,
                          const RendererSettings& settings);
    ~NativeRenderExtension();
    NativeRenderExtension(const NativeRenderExtension&) = delete;
    NativeRenderExtension& operator=(const NativeRenderExtension&) = delete;

    Rasterizer& rasterizer() { return *_rasterizer; }
    const Rasterizer& rasterizer() const { return *_rasterizer; }
    void setViewportSize(int width, int height);
    std::unique_ptr<Backend::Texture>
    createTexture(const Scene::TextureSourceResource& source);
    std::unique_ptr<Backend::GraphicsPipeline>
    createSceneHookPipeline(const SceneHookPipelineDesc& desc);
    // The facade synchronizes scene sources before this draw.
    void renderSceneToFramebuffer(const glm::mat4& view,
                                  const glm::mat4& projection,
                                  Backend::Framebuffer* target, int width,
                                  int height, bool clear);

  private:
    void ensureOffscreenSceneTarget(Backend::Framebuffer* target, int width,
                                    int height);
    void ensureOffscreenClearTarget(const glm::vec4& clearColor);
    void ensureOffscreenFormatConversion(Backend::Texture* targetColor);
    void recordOffscreenFormatConversion(int width, int height);
    Backend::GraphicsDevice* _device;
    Rasterizer* _rasterizer;
    const RendererSettings& _settings;
    int _viewportWidth = 0;
    int _viewportHeight = 0;
    Backend::Framebuffer* _offscreenFramebuffer = nullptr;
    Backend::Texture* _offscreenResolvedColor = nullptr;
    int _offscreenWidth = 0;
    int _offscreenHeight = 0;
    std::unique_ptr<Backend::Texture> _offscreenMsaaColor;
    std::unique_ptr<Backend::Texture> _offscreenIntermediateColor;
    std::unique_ptr<Backend::Texture> _offscreenMsaaDepthStencil;
    std::unique_ptr<Backend::TextureView> _offscreenMsaaColorView;
    std::unique_ptr<Backend::TextureView> _offscreenResolveColorView;
    std::unique_ptr<Backend::TextureView> _offscreenDepthStencilView;
    std::unique_ptr<Backend::RenderTarget> _offscreenClearTarget;
    std::unique_ptr<Backend::RenderTarget> _offscreenDrawTarget;
    std::unique_ptr<Backend::RenderTarget> _offscreenResolveTarget;
    std::unique_ptr<Backend::TextureView> _offscreenOutputView;
    std::unique_ptr<Backend::RenderTarget> _offscreenOutputTarget;
    std::array<std::unique_ptr<Backend::BindGroupLayout>, 4>
        _offscreenCopyGroupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _offscreenCopyPipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _offscreenCopyPipeline;
    std::unique_ptr<Backend::Sampler> _offscreenCopySampler;
    std::unique_ptr<Backend::BindGroup> _offscreenCopyBindGroup;
    Backend::TextureFormat _offscreenOutputFormat =
        Backend::TextureFormat::Undefined;
    std::unique_ptr<Backend::PipelineLayout> _sceneHookDefaultPipelineLayout;
    std::unique_ptr<Backend::PipelineLayout> _sceneHookFramePipelineLayout;
};

} // namespace KE

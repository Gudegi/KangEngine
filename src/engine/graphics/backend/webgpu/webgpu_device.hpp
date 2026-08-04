#ifndef _WEBGPU_DEVICE_HPP_
#define _WEBGPU_DEVICE_HPP_

#include "../base/graphics_device.hpp"

namespace KE {
namespace Backend {

// Minimal WebGPU backend placeholder.
//
// This keeps BackendType::WebGPU represented by a real GraphicsDevice while
// resource creation and drawing are still intentionally unimplemented.
class WebGPUDevice : public GraphicsDevice {
  private:
    bool _initialized = false;

  public:
    WebGPUDevice() = default;
    ~WebGPUDevice() override;

    void initialize() override;
    void shutdown() override;
    BackendType getBackendType() const override { return BackendType::WebGPU; }

    void beginFrame() override;
    void endFrame() override;
    void clear(float r, float g, float b, float a) override;
    void setViewport(int x, int y, int width, int height) override;
    void drawIndexed(size_t indexCount) override;
    void drawLines(size_t vertexCount) override;
    void drawPoints(size_t vertexCount) override;
    void drawIndexedInstanced(size_t indexCount, size_t instanceCount) override;
    void checkError() override;

    void setDepthTest(bool enable) override;
    void setDepthWrite(bool enable) override;
    void setColorWrite(bool enable) override;
    void setBlend(bool enable) override;
    void setBlendFunc(BlendFactor src, BlendFactor dst) override;
    void setStencilTest(bool enable) override;
    void setStencilFunc(StencilFunc func, int ref, uint32_t mask) override;
    void setStencilOp(StencilOp stencilFail, StencilOp depthFail,
                      StencilOp depthPass) override;
    void setStencilWriteMask(uint32_t mask) override;
    void setPolygonMode(PolygonMode mode) override;
    void setLineWidth(float width) override;
    void setCullFace(bool enable) override;
    void setCullFaceMode(CullFaceMode mode) override;
    void setClearColor(float r, float g, float b, float a) override;

    std::unique_ptr<Buffer> createBuffer(const BufferDesc& desc,
                                         const void* data = nullptr) override;
    void bindUniformBuffer(Buffer* buffer, int slot) override;
    std::unique_ptr<Texture> createTexture(const TextureDesc& desc) override;
    std::unique_ptr<Texture>
    createTexture(const TextureResourceDesc& desc,
                  const TextureInitialData* initialData = nullptr) override;
    std::unique_ptr<TextureView>
    createTextureView(Texture* texture,
                      const TextureViewDesc& desc = {}) override;
    std::unique_ptr<Sampler>
    createSampler(const SamplerDesc& desc = {}) override;
    std::unique_ptr<RenderTarget>
    createRenderTarget(const RenderPassDesc& desc) override;
    std::unique_ptr<GraphicsPipeline>
    createGraphicsPipeline(const GraphicsPipelineDesc& desc) override;
    std::unique_ptr<BindGroupLayout>
    createBindGroupLayout(const BindGroupLayoutDesc& desc) override;
    std::unique_ptr<PipelineLayout>
    createPipelineLayout(const PipelineLayoutDesc& desc) override;
    std::unique_ptr<BindGroup>
    createBindGroup(const BindGroupDesc& desc) override;
    TextureReadback readTexture(TextureView* view) override;
    std::unique_ptr<CommandEncoder> createCommandEncoder() override;
    void submit(CommandBuffer& commandBuffer) override;

    std::unique_ptr<Texture> createTexture(const std::string path,
                                           bool flip = false) override;
    std::unique_ptr<Texture> createTexture(const std::string path, bool flip,
                                           const SamplerDesc& sampler) override;
    std::unique_ptr<Texture> createTexture(const std::string path, bool flip,
                                           float warpParam,
                                           float minFilferParam,
                                           float maxFilterParam) override;

    std::unique_ptr<Framebuffer>
    createFramebuffer(const FramebufferDesc& desc) override;
};

} // namespace Backend
} // namespace KE

#endif

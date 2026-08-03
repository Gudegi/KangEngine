#include "webgpu_device.hpp"

#include <iostream>
#include <stdexcept>

namespace KE {
namespace Backend {

namespace {

[[noreturn]] void throwNotImplemented(const char* functionName) {
    throw std::runtime_error(std::string("WebGPUDevice::") + functionName +
                             " is not implemented yet");
}

} // namespace

WebGPUDevice::~WebGPUDevice() {
    if (_initialized)
        shutdown();
}

void WebGPUDevice::initialize() {
    if (_initialized)
        return;
    _initialized = true;
    std::cout << "WebGPU Device skeleton initialized" << std::endl;
}

void WebGPUDevice::shutdown() {
    if (!_initialized)
        return;
    _initialized = false;
    std::cout << "WebGPU Device skeleton shutdown" << std::endl;
}

void WebGPUDevice::beginFrame() { throwNotImplemented("beginFrame"); }

void WebGPUDevice::endFrame() { throwNotImplemented("endFrame"); }

void WebGPUDevice::clear(float, float, float, float) {
    throwNotImplemented("clear");
}

void WebGPUDevice::setViewport(int, int, int, int) {
    throwNotImplemented("setViewport");
}

void WebGPUDevice::drawIndexed(size_t) { throwNotImplemented("drawIndexed"); }

void WebGPUDevice::drawLines(size_t) { throwNotImplemented("drawLines"); }

void WebGPUDevice::drawPoints(size_t) { throwNotImplemented("drawPoints"); }

void WebGPUDevice::drawIndexedInstanced(size_t, size_t) {
    throwNotImplemented("drawIndexedInstanced");
}

void WebGPUDevice::checkError() {}

void WebGPUDevice::setDepthTest(bool) { throwNotImplemented("setDepthTest"); }

void WebGPUDevice::setDepthWrite(bool) { throwNotImplemented("setDepthWrite"); }

void WebGPUDevice::setColorWrite(bool) { throwNotImplemented("setColorWrite"); }

void WebGPUDevice::setBlend(bool) { throwNotImplemented("setBlend"); }

void WebGPUDevice::setBlendFunc(BlendFactor, BlendFactor) {
    throwNotImplemented("setBlendFunc");
}

void WebGPUDevice::setStencilTest(bool) {
    throwNotImplemented("setStencilTest");
}

void WebGPUDevice::setStencilFunc(StencilFunc, int, uint32_t) {
    throwNotImplemented("setStencilFunc");
}

void WebGPUDevice::setStencilOp(StencilOp, StencilOp, StencilOp) {
    throwNotImplemented("setStencilOp");
}

void WebGPUDevice::setStencilWriteMask(uint32_t) {
    throwNotImplemented("setStencilWriteMask");
}

void WebGPUDevice::setPolygonMode(PolygonMode) {
    throwNotImplemented("setPolygonMode");
}

void WebGPUDevice::setLineWidth(float) { throwNotImplemented("setLineWidth"); }

void WebGPUDevice::setCullFace(bool) { throwNotImplemented("setCullFace"); }

void WebGPUDevice::setCullFaceMode(CullFaceMode) {
    throwNotImplemented("setCullFaceMode");
}

void WebGPUDevice::setClearColor(float, float, float, float) {
    throwNotImplemented("setClearColor");
}

std::unique_ptr<Buffer> WebGPUDevice::createBuffer(const BufferDesc&,
                                                    const void*) {
    throwNotImplemented("createBuffer(BufferDesc)");
}

void WebGPUDevice::bindUniformBuffer(Buffer*, int) {
    throwNotImplemented("bindUniformBuffer");
}

std::unique_ptr<Shader> WebGPUDevice::createShader(const ShaderDesc&) {
    throwNotImplemented("createShader");
}

std::unique_ptr<Texture> WebGPUDevice::createTexture(const TextureDesc&) {
    throwNotImplemented("createTexture");
}

std::unique_ptr<Texture>
WebGPUDevice::createTexture(const TextureResourceDesc&,
                            const TextureInitialData*) {
    throwNotImplemented("createTexture(TextureResourceDesc)");
}

std::unique_ptr<TextureView>
WebGPUDevice::createTextureView(Texture*, const TextureViewDesc&) {
    throwNotImplemented("createTextureView");
}

std::unique_ptr<Sampler> WebGPUDevice::createSampler(const SamplerDesc&) {
    throwNotImplemented("createSampler");
}

std::unique_ptr<RenderTarget>
WebGPUDevice::createRenderTarget(const RenderPassDesc&) {
    throwNotImplemented("createRenderTarget");
}

std::unique_ptr<GraphicsPipeline>
WebGPUDevice::createGraphicsPipeline(const GraphicsPipelineDesc&) {
    throwNotImplemented("createGraphicsPipeline");
}

std::unique_ptr<BindGroupLayout>
WebGPUDevice::createBindGroupLayout(const BindGroupLayoutDesc&) {
    throwNotImplemented("createBindGroupLayout");
}
std::unique_ptr<PipelineLayout>
WebGPUDevice::createPipelineLayout(const PipelineLayoutDesc&) {
    throwNotImplemented("createPipelineLayout");
}
std::unique_ptr<BindGroup> WebGPUDevice::createBindGroup(const BindGroupDesc&) {
    throwNotImplemented("createBindGroup");
}
TextureReadback WebGPUDevice::readTexture(TextureView*) {
    throwNotImplemented("readTexture");
}

std::unique_ptr<CommandEncoder> WebGPUDevice::createCommandEncoder() {
    throwNotImplemented("createCommandEncoder");
}

void WebGPUDevice::submit(CommandBuffer&) { throwNotImplemented("submit"); }

std::unique_ptr<Shader> WebGPUDevice::createShader(const char*, const char*) {
    throwNotImplemented("createShader");
}

std::unique_ptr<Shader> WebGPUDevice::createShader(const std::string&,
                                                   const std::string&) {
    throwNotImplemented("createShader");
}

std::unique_ptr<Texture> WebGPUDevice::createTexture(const std::string, bool) {
    throwNotImplemented("createTexture");
}

std::unique_ptr<Texture> WebGPUDevice::createTexture(const std::string, bool,
                                                     const SamplerDesc&) {
    throwNotImplemented("createTexture");
}

std::unique_ptr<Texture> WebGPUDevice::createTexture(const std::string, bool,
                                                     float, float, float) {
    throwNotImplemented("createTexture");
}

std::unique_ptr<Framebuffer>
WebGPUDevice::createFramebuffer(const FramebufferDesc&) {
    throwNotImplemented("createFramebuffer");
}

} // namespace Backend
} // namespace KE

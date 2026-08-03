#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"

#include <iostream>
#include <stdexcept>

using namespace KE;
using namespace KE::Backend;

namespace {
constexpr const char* FullscreenTriangleVs = R"(
#version 410 core
void main() {
    const vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0),
                              vec2(-1.0, 3.0));
    gl_Position = vec4(p[gl_VertexID], 0.0, 1.0);
}
)";

constexpr const char* ColorProducerFs = R"(
#version 410 core
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(0.15, 0.25, 0.35, 1.0); }
)";

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

int main() {
    Window window;
    window.init(32, 32, true);
    require(window.getGlfwWindow() != nullptr, "hidden window creation failed");
    OpenGLDevice device;
    device.initialize();
    device.setValidationEnabled(true);

    FramebufferDesc legacyDesc;
    legacyDesc.width = 8;
    legacyDesc.height = 8;
    auto legacy = device.createFramebuffer(legacyDesc);
    legacy->bind();
    device.clear(0.8f, 0.1f, 0.2f, 1.0f);
    legacy->unbind();

    TextureResourceDesc textureDesc;
    textureDesc.extent = {8, 8, 1};
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.usage = TextureUsage::RenderAttachment | TextureUsage::CopySrc;
    textureDesc.label = "coexist_rhi_color";
    auto texture = device.createTexture(textureDesc);
    TextureViewDesc viewDesc;
    viewDesc.format = TextureFormat::RGBA8Unorm;
    auto view = device.createTextureView(texture.get(), viewDesc);
    RenderPassDesc passDesc;
    passDesc.label = "coexist_rhi_pass";
    passDesc.colorAttachments.push_back(
        {view.get(), nullptr, LoadOp::Clear, StoreOp::Store,
         {0.1f, 0.2f, 0.8f, 1.0f}});
    auto target = device.createRenderTarget(passDesc);
    GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "coexist_rhi_pipeline";
    pipelineDesc.primitive.cullMode = CullMode::None;
    pipelineDesc.colorTargets = {{TextureFormat::RGBA8Unorm}};
    pipelineDesc.shader.stages = {
        {FullscreenTriangleVs, ShaderType::Vertex, "main"},
        {ColorProducerFs, ShaderType::Fragment, "main"}};
    auto pipeline = device.createGraphicsPipeline(pipelineDesc);
    auto encoder = device.createCommandEncoder();
    auto pass = encoder->beginRenderPass(target.get());
    pass->setPipeline(pipeline.get());
    pass->draw(3);
    pass->end();
    auto commands = encoder->finish();
    device.submit(*commands);

    legacy->bind();
    device.clear(0.1f, 0.8f, 0.2f, 1.0f);
    legacy->unbind();
    const auto pixels = legacy->readColorPixels(false);
    require(pixels.size() >= 4 && pixels[0] < 40 && pixels[1] > 190 &&
                pixels[2] < 70,
            "legacy framebuffer state was not usable after RHI submission");
    const auto rhiPixels = device.readTexture(view.get());
    require(!rhiPixels.values.empty() && rhiPixels.values[2] > 0.3f,
            "RHI output was lost during legacy coexistence test");
    std::cout << "PASS: legacy forward framebuffer before/after RHI submission"
              << std::endl;
    return 0;
}

#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"
#include "engine/graphics/renderer/post_processor.hpp"

#include <cmath>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace KE;
using namespace KE::Backend;

namespace {
struct RenderDocApi100Prefix {
    void* entriesBeforeStartFrameCapture[19];
    void (*startFrameCapture)(void*, void*);
    uint32_t (*isFrameCapturing)();
    uint32_t (*endFrameCapture)(void*, void*);
};

RenderDocApi100Prefix* beginRenderDocCaptureIfInjected() {
    using GetApi = int (*)(int, void**);
    auto getApi = reinterpret_cast<GetApi>(
        dlsym(RTLD_DEFAULT, "RENDERDOC_GetAPI"));
    if (!getApi)
        return nullptr;
    RenderDocApi100Prefix* api = nullptr;
    if (getApi(10000, reinterpret_cast<void**>(&api)) == 1 && api &&
        api->startFrameCapture) {
        api->startFrameCapture(nullptr, nullptr);
        return api;
    }
    return nullptr;
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void expectNear(float actual, float expected, float tolerance,
                const char* message) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message);
}

std::unique_ptr<Texture> makeHdrScene(GraphicsDevice& device, uint32_t width,
                                      uint32_t height,
                                      const ClearColor& color) {
    TextureResourceDesc desc;
    desc.extent = {width, height, 1};
    desc.format = TextureFormat::RGBA16Float;
    desc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding |
                 TextureUsage::CopySrc;
    desc.label = "post_smoke_hdr_scene";
    auto texture = device.createTexture(desc);
    TextureViewDesc viewDesc;
    viewDesc.format = desc.format;
    auto view = device.createTextureView(texture.get(), viewDesc);
    RenderPassDesc passDesc;
    passDesc.label = "post_smoke_hdr_clear";
    passDesc.colorAttachments = {
        {view.get(), nullptr, LoadOp::Clear, StoreOp::Store, color}};
    auto target = device.createRenderTarget(passDesc);
    auto encoder = device.createCommandEncoder();
    auto pass = encoder->beginRenderPass(target.get());
    pass->end();
    auto commands = encoder->finish();
    device.submit(*commands);
    return texture;
}

TextureReadback readOutput(GraphicsDevice& device, PostProcessor& post) {
    TextureViewDesc desc;
    desc.format = TextureFormat::RGBA8Unorm;
    auto view = device.createTextureView(post.getResult(), desc);
    return device.readTexture(view.get());
}
} // namespace

int main(int argc, char** argv) {
    const bool captureVisible =
        argc > 1 && std::string(argv[1]) == "--capture-visible";
    Window window;
    window.init(32, 32, !captureVisible);
    require(window.getGlfwWindow() != nullptr, "hidden window creation failed");
    OpenGLDevice device;
    device.initialize();
    device.setValidationEnabled(true);
    auto* renderDocApi =
        captureVisible ? beginRenderDocCaptureIfInjected() : nullptr;

    PostProcessor post;
    post.init(&device, 8, 8);
    BloomConfig bloom;
    bloom.enabled = false;
    auto scene = makeHdrScene(device, 8, 8, {2.0f, 1.0f, 0.5f, 1.0f});
    post.process(scene.get(), 1.0f, ToneMapMode::Reinhard, 1.0f, bloom);
    auto pixels = readOutput(device, post);
    require(pixels.width == 8 && pixels.height == 8,
            "tone-map output extent mismatch");
    expectNear(pixels.values[0], 2.0f / 3.0f, 0.01f,
               "Reinhard red mismatch");
    expectNear(pixels.values[1], 0.5f, 0.01f, "Reinhard green mismatch");
    expectNear(pixels.values[2], 1.0f / 3.0f, 0.01f,
               "Reinhard blue mismatch");

    bloom.enabled = true;
    bloom.threshold = 0.5f;
    bloom.intensity = 0.25f;
    // A uniform source remains unchanged while exercising both horizontal and
    // vertical ping-pong blur passes.
    bloom.iterations = 2;
    bloom.downsample = 1;
    scene = makeHdrScene(device, 8, 8, {0.8f, 0.2f, 0.1f, 1.0f});
    post.process(scene.get(), 1.0f, ToneMapMode::None, 1.0f, bloom);
    pixels = readOutput(device, post);
    expectNear(pixels.values[0], 1.0f, 0.01f,
               "bright-extract bloom red mismatch");
    expectNear(pixels.values[1], 0.25f, 0.01f,
               "bright-extract bloom green mismatch");
    expectNear(pixels.values[2], 0.125f, 0.01f,
               "bright-extract bloom blue mismatch");

    bloom.downsample = 2;
    bloom.iterations = 3;
    post.process(scene.get(), 1.0f, ToneMapMode::None, 1.0f, bloom);
    pixels = readOutput(device, post);
    expectNear(pixels.values[0], 1.0f, 0.01f,
               "resized ping-pong bloom red mismatch");
    expectNear(pixels.values[1], 0.25f, 0.01f,
               "resized ping-pong bloom green mismatch");

    post.resize(11, 7);
    bloom.enabled = false;
    scene = makeHdrScene(device, 11, 7, {0.25f, 0.25f, 0.25f, 1.0f});
    post.process(scene.get(), 2.0f, ToneMapMode::None, 1.0f, bloom);
    pixels = readOutput(device, post);
    require(pixels.width == 11 && pixels.height == 7,
            "resized tone-map output extent mismatch");
    expectNear(pixels.values[0], 0.5f, 0.01f, "gamma correction mismatch");

    post.resize(0, 0);
    post.process(scene.get(), 1.0f, ToneMapMode::None, 1.0f, bloom);
    glfwSwapBuffers(window.getGlfwWindow());
    if (renderDocApi)
        require(renderDocApi->endFrameCapture(nullptr, nullptr) == 1,
                "RenderDoc failed to save the PostProcessor capture");
    std::cout << "PASS: complete production PostProcessor uses RHI extract, "
                 "ping-pong blur, composite, tone-map/gamma, resize, and "
                 "zero-size skip"
              << std::endl;
    return 0;
}

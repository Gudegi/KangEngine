#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"
#include "engine/graphics/renderer/selection_outline_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace KE;
using namespace KE::Backend;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::unique_ptr<Texture> makeScene(GraphicsDevice& device, uint32_t width,
                                   uint32_t height) {
    TextureResourceDesc desc;
    desc.extent = {width, height, 1};
    desc.format = TextureFormat::RGBA16Float;
    desc.usage = TextureUsage::RenderAttachment | TextureUsage::TextureBinding |
                 TextureUsage::CopySrc;
    desc.label = "selection_smoke_scene";
    auto texture = device.createTexture(desc);
    TextureViewDesc viewDesc;
    viewDesc.format = desc.format;
    auto view = device.createTextureView(texture.get(), viewDesc);
    RenderPassDesc passDesc;
    passDesc.label = "selection_smoke_scene_clear";
    passDesc.colorAttachments = {
        {view.get(), nullptr, LoadOp::Clear, StoreOp::Store,
         {0.2f, 0.3f, 0.4f, 1.0f}}};
    auto target = device.createRenderTarget(passDesc);
    auto encoder = device.createCommandEncoder();
    auto pass = encoder->beginRenderPass(target.get());
    pass->end();
    auto commands = encoder->finish();
    device.submit(*commands);
    return texture;
}

std::unique_ptr<Texture> makeMask(GraphicsDevice& device, uint32_t width,
                                  uint32_t height) {
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4, 0);
    const uint32_t cx = width / 2;
    const uint32_t cy = height / 2;
    for (uint32_t y = cy; y < std::min(height, cy + 2); ++y) {
        for (uint32_t x = cx; x < std::min(width, cx + 2); ++x) {
            const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
            pixels[offset] = 255;
            pixels[offset + 3] = 255;
        }
    }
    TextureResourceDesc desc;
    desc.extent = {width, height, 1};
    desc.format = TextureFormat::RGBA8Unorm;
    desc.usage = TextureUsage::TextureBinding | TextureUsage::CopySrc |
                 TextureUsage::CopyDst;
    desc.label = "selection_smoke_mask";
    TextureInitialData initial;
    initial.data = pixels.data();
    initial.size = pixels.size();
    initial.bytesPerRow = static_cast<size_t>(width) * 4;
    return device.createTexture(desc, &initial);
}

void verifyOutput(GraphicsDevice& device, SelectionOutlineProcessor& outline,
                  uint32_t width, uint32_t height) {
    TextureViewDesc outputDesc;
    outputDesc.format = TextureFormat::RGBA16Float;
    auto outputView =
        device.createTextureView(outline.getResult(), outputDesc);
    const TextureReadback readback = device.readTexture(outputView.get());
    require(readback.width == width && readback.height == height,
            "selection output extent mismatch");
    bool foundScene = false;
    bool foundOutline = false;
    for (size_t i = 0; i + 3 < readback.values.size(); i += 4) {
        const float r = readback.values[i];
        const float g = readback.values[i + 1];
        const float b = readback.values[i + 2];
        foundScene |= std::abs(r - 0.2f) < 0.02f &&
                      std::abs(g - 0.3f) < 0.02f &&
                      std::abs(b - 0.4f) < 0.02f;
        foundOutline |= r > 0.8f && g < 0.1f && b < 0.1f;
    }
    require(foundScene, "selection composite lost the original scene color");
    require(foundOutline, "selection composite did not produce an outline");
}
} // namespace

int main() {
    Window window;
    window.init(32, 32, true);
    require(window.getGlfwWindow() != nullptr, "hidden window creation failed");
    OpenGLDevice device;
    device.initialize();
    device.setValidationEnabled(true);

    SelectionOutlineProcessor outline;
    outline.init(&device, 8, 8);
    outline.setOutlineColor({1.0f, 0.0f, 0.0f, 1.0f});
    outline.setOutlineRadius(1.0f);
    auto scene = makeScene(device, 8, 8);
    auto mask = makeMask(device, 8, 8);
    outline.renderOutlineCompositePass(scene.get(), mask.get());
    verifyOutput(device, outline, 8, 8);

    outline.resize(11, 7);
    scene = makeScene(device, 11, 7);
    mask = makeMask(device, 11, 7);
    outline.renderOutlineCompositePass(scene.get(), mask.get());
    verifyOutput(device, outline, 11, 7);
    outline.resize(0, 0);
    outline.renderOutlineCompositePass(scene.get(), mask.get());

    std::cout << "PASS: production selection outline uses RHI fullscreen pass, "
                 "bindings, resize, and zero-size skip"
              << std::endl;
    return 0;
}

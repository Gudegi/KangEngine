#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace KE;
using namespace KE::Backend;

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void expectLogicError(const std::function<void()>& operation,
                      const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, message);
}

struct TargetFixture {
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    std::unique_ptr<RenderTarget> target;
};

TargetFixture makeTarget(OpenGLDevice& device, const char* label,
                         const ClearColor& clear) {
    TextureResourceDesc textureDesc;
    textureDesc.extent = {8, 8, 1};
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.usage = TextureUsage::RenderAttachment |
                        TextureUsage::TextureBinding;
    textureDesc.label = std::string(label) + "_texture";

    TargetFixture fixture;
    fixture.texture = device.createTexture(textureDesc);
    TextureViewDesc viewDesc;
    viewDesc.format = TextureFormat::RGBA8Unorm;
    viewDesc.label = std::string(label) + "_view";
    fixture.view = device.createTextureView(fixture.texture.get(), viewDesc);
    RenderPassDesc passDesc;
    passDesc.label = label;
    passDesc.colorAttachments.push_back(
        {fixture.view.get(), nullptr, LoadOp::Clear, StoreOp::Store, clear});
    fixture.target = device.createRenderTarget(passDesc);
    return fixture;
}

std::array<float, 4> readPixel(RenderTarget* target) {
    auto* glTarget = dynamic_cast<OpenGLRenderTarget*>(target);
    require(glTarget != nullptr, "target is not OpenGL-backed");
    std::array<float, 4> pixel{};
    glBindFramebuffer(GL_READ_FRAMEBUFFER, glTarget->getHandle());
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_FLOAT, pixel.data());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    return pixel;
}

} // namespace

int main() {
    Window window;
    window.init(64, 64, true);
    require(window.getGlfwWindow() != nullptr,
            "failed to create hidden command smoke window");

    OpenGLDevice device;
    device.initialize();
    auto first = makeTarget(device, "command_first",
                            {0.2f, 0.4f, 0.6f, 1.0f});
    auto second = makeTarget(device, "command_second",
                             {0.8f, 0.3f, 0.1f, 1.0f});

    auto encoder = device.createCommandEncoder();
    auto firstPass = encoder->beginRenderPass(first.target.get());
    firstPass->setViewport(0, 0, 8, 8);
    firstPass->setScissor(0, 0, 8, 8);
    expectLogicError(
        [&] { (void)encoder->beginRenderPass(second.target.get()); },
        "nested render pass was accepted");
    expectLogicError([&] { (void)encoder->finish(); },
                     "encoder finished with an active pass");
    firstPass->end();
    expectLogicError([&] { firstPass->end(); },
                     "render pass ended twice");

    auto secondPass = encoder->beginRenderPass(second.target.get());
    secondPass->setViewport(0, 0, 8, 8, 0.1f, 0.9f);
    secondPass->end();
    auto commands = encoder->finish();
    expectLogicError(
        [&] { (void)encoder->beginRenderPass(first.target.get()); },
        "finished encoder accepted another pass");
    expectLogicError([&] { (void)encoder->finish(); },
                     "encoder finished twice");

    glViewport(7, 9, 31, 27);
    glEnable(GL_SCISSOR_TEST);
    glScissor(2, 3, 17, 19);
    device.submit(*commands);
    GLint viewport[4] = {};
    GLint scissor[4] = {};
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetIntegerv(GL_SCISSOR_BOX, scissor);
    require(viewport[0] == 7 && viewport[1] == 9 && viewport[2] == 31 &&
                viewport[3] == 27,
            "submit did not restore legacy viewport state");
    require(glIsEnabled(GL_SCISSOR_TEST) && scissor[0] == 2 &&
                scissor[1] == 3 && scissor[2] == 17 && scissor[3] == 19,
            "submit did not restore legacy scissor state");
    expectLogicError([&] { device.submit(*commands); },
                     "command buffer was submitted twice");

    const auto firstPixel = readPixel(first.target.get());
    const auto secondPixel = readPixel(second.target.get());
    require(std::abs(firstPixel[0] - 0.2f) < 0.01f &&
                std::abs(firstPixel[2] - 0.6f) < 0.01f,
            "first encoded pass result mismatch");
    require(std::abs(secondPixel[0] - 0.8f) < 0.01f &&
                std::abs(secondPixel[2] - 0.1f) < 0.01f,
            "second encoded pass result mismatch");

    auto threadEncoder = device.createCommandEncoder();
    auto threadPass = threadEncoder->beginRenderPass(first.target.get());
    threadPass->end();
    auto threadCommands = threadEncoder->finish();
    bool rejectedWrongThread = false;
    std::thread worker([&] {
        try {
            device.submit(*threadCommands);
        } catch (const std::runtime_error&) {
            rejectedWrongThread = true;
        }
    });
    worker.join();
    require(rejectedWrongThread,
            "OpenGL submission from a worker thread was accepted");

    std::unique_ptr<CommandBuffer> workerRecordedCommands;
    std::thread recorder([&] {
        auto workerEncoder = device.createCommandEncoder();
        auto workerPass =
            workerEncoder->beginRenderPass(first.target.get());
        workerPass->setViewport(0, 0, 8, 8);
        workerPass->end();
        workerRecordedCommands = workerEncoder->finish();
    });
    recorder.join();
    require(workerRecordedCommands != nullptr,
            "worker failed to record a command buffer");
    device.submit(*workerRecordedCommands);

    glDisable(GL_SCISSOR_TEST);
    require(glGetError() == GL_NO_ERROR,
            "OpenGL error in command encoder smoke test");
    std::cout << "PASS: recorded render passes, submission, and state validation"
              << std::endl;
    return 0;
}

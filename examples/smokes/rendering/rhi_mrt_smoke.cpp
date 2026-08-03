#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

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

constexpr const char* IndexedTriangleVs = R"(
#version 410 core
layout(location = 0) in vec2 inPosition;
void main() { gl_Position = vec4(inPosition, 0.0, 1.0); }
)";

constexpr const char* MrtProducerFs = R"(
#version 410 core
layout(location = 0) out vec4 g0;
layout(location = 1) out vec4 g1;
layout(location = 2) out vec4 g2;
void main() {
    g0 = vec4(0.2, 0.4, 0.6, 1.0);
    g1 = vec4(1.25, 2.5, 3.75, 1.0);
    g2 = vec4(0.8, 0.3, 0.1, 1.0);
}
)";

constexpr const char* GbufferConsumerFs = R"(
#version 410 core
uniform sampler2D ke_g3_b0;
uniform sampler2D ke_g3_b1;
uniform sampler2D ke_g3_b2;
uniform sampler2D ke_g3_b5;
layout(std140) uniform ke_g3_b4 { vec4 weights; };
layout(location = 0) out vec4 outColor;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 a = texelFetch(ke_g3_b0, p, 0);
    vec4 b = texelFetch(ke_g3_b1, p, 0);
    vec4 c = texelFetch(ke_g3_b2, p, 0);
    float d = texelFetch(ke_g3_b5, p, 0).r;
    outColor = a * weights.x + b * weights.y + c * weights.z +
               vec4(d) * weights.w;
}
)";

constexpr const char* DepthOnlyFs = R"(
#version 410 core
void main() {}
)";

constexpr const char* ColorProducerFs = R"(
#version 410 core
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(0.15, 0.25, 0.35, 1.0); }
)";

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

TextureResourceDesc textureDesc(uint32_t width, uint32_t height,
                                TextureFormat format, TextureUsage usage,
                                const char* label) {
    TextureResourceDesc desc;
    desc.extent = {width, height, 1};
    desc.format = format;
    desc.usage = usage;
    desc.label = label;
    return desc;
}

std::unique_ptr<TextureView> makeView(OpenGLDevice& device, Texture* texture,
                                      TextureFormat format,
                                      TextureAspect aspect, const char* label) {
    TextureViewDesc desc;
    desc.format = format;
    desc.aspect = aspect;
    desc.label = label;
    return device.createTextureView(texture, desc);
}

void expectInvalid(const std::function<void()>& operation,
                   const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

std::array<float, 4> readPixel(GraphicsDevice& device, TextureView* view) {
    const TextureReadback readback = device.readTexture(view);
    require(readback.componentCount > 0 && !readback.values.empty(),
            "texture readback is empty");
    const uint32_t x = std::min(3u, readback.width - 1);
    const uint32_t y = std::min(3u, readback.height - 1);
    const size_t base =
        (static_cast<size_t>(y) * readback.width + x) *
        readback.componentCount;
    std::array<float, 4> pixel{0.0f, 0.0f, 0.0f, 1.0f};
    for (uint32_t channel = 0;
         channel < std::min(4u, readback.componentCount); ++channel)
        pixel[channel] = readback.values[base + channel];
    return pixel;
}

void expectNear(float actual, float expected, float tolerance,
                const char* message) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(std::string(message) + " (actual=" +
                                 std::to_string(actual) + ", expected=" +
                                 std::to_string(expected) + ")");
}

} // namespace

int main(int argc, char** argv) {
    const bool captureVisible =
        argc > 1 && std::string(argv[1]) == "--capture-visible";
    Window window;
    window.init(64, 64, !captureVisible);
    require(window.getGlfwWindow() != nullptr,
            "failed to create hidden OpenGL MRT smoke window");

    OpenGLDevice device;
    device.initialize();
    device.setValidationEnabled(true);
    auto* renderDocApi =
        captureVisible ? beginRenderDocCaptureIfInjected() : nullptr;

    constexpr uint32_t width = 8;
    constexpr uint32_t height = 8;
    constexpr TextureUsage attachmentUsage =
        TextureUsage::RenderAttachment | TextureUsage::TextureBinding |
        TextureUsage::CopySrc;

    auto color0 = device.createTexture(textureDesc(
        width, height, TextureFormat::RGBA8Unorm, attachmentUsage, "mrt_g0"));
    auto color1 = device.createTexture(textureDesc(
        width, height, TextureFormat::RGBA16Float, attachmentUsage, "mrt_g1"));
    auto color2 = device.createTexture(textureDesc(
        width, height, TextureFormat::RGBA8Unorm, attachmentUsage, "mrt_g2"));
    auto depth = device.createTexture(
        textureDesc(width, height, TextureFormat::Depth32Float,
        TextureUsage::RenderAttachment | TextureUsage::TextureBinding |
            TextureUsage::CopySrc,
        "mrt_depth"));

    auto color0View = makeView(device, color0.get(), TextureFormat::RGBA8Unorm,
                               TextureAspect::All, "mrt_g0_view");
    auto color1View = makeView(device, color1.get(), TextureFormat::RGBA16Float,
                               TextureAspect::All, "mrt_g1_view");
    auto color2View = makeView(device, color2.get(), TextureFormat::RGBA8Unorm,
                               TextureAspect::All, "mrt_g2_view");
    auto depthView = makeView(device, depth.get(), TextureFormat::Depth32Float,
                              TextureAspect::DepthOnly, "mrt_depth_view");

    RenderPassDesc passDesc;
    passDesc.label = "rhi_mrt_smoke";
    passDesc.colorAttachments = {
        {color0View.get(),
         nullptr,
         LoadOp::Clear,
         StoreOp::Store,
         {0.01f, 0.01f, 0.01f, 1.0f}},
        {color1View.get(),
         nullptr,
         LoadOp::Clear,
         StoreOp::Store,
         {0.02f, 0.02f, 0.02f, 1.0f}},
        {color2View.get(),
         nullptr,
         LoadOp::Clear,
         StoreOp::Store,
         {0.03f, 0.03f, 0.03f, 1.0f}},
    };
    passDesc.depthStencilAttachment =
        DepthStencilAttachmentDesc{depthView.get(),
                                   LoadOp::Clear,
                                   StoreOp::Store,
                                   1.0f,
                                   LoadOp::Load,
                                   StoreOp::Store,
                                   0};

    auto target = device.createRenderTarget(passDesc);
    require(target->getWidth() == width && target->getHeight() == height &&
                target->getSampleCount() == 1,
            "MRT target metadata mismatch");

    GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "rhi_mrt_pipeline";
    pipelineDesc.shader.name = "rhi_mrt_shader";
    pipelineDesc.shader.stages = {
        {IndexedTriangleVs, ShaderType::Vertex, "main"},
        {MrtProducerFs, ShaderType::Fragment, "main"},
    };
    pipelineDesc.primitive.cullMode = CullMode::None;
    VertexBufferLayout mrtPositionLayout;
    mrtPositionLayout.arrayStride = sizeof(float) * 2;
    mrtPositionLayout.attributes.push_back(
        {VertexFormat::Float32x2, 0, 0});
    pipelineDesc.vertexBuffers = {mrtPositionLayout};
    pipelineDesc.depthStencil = DepthStencilState{
        TextureFormat::Depth32Float, true, CompareFunction::Less};
    pipelineDesc.colorTargets = {{TextureFormat::RGBA8Unorm},
                                 {TextureFormat::RGBA16Float},
                                 {TextureFormat::RGBA8Unorm}};
    auto pipeline = device.createGraphicsPipeline(pipelineDesc);
    const std::array<float, 6> mrtPositions = {
        -1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    const std::array<uint16_t, 3> mrtIndices = {0, 1, 2};
    BufferDesc mrtVertexDesc;
    mrtVertexDesc.size = sizeof(mrtPositions);
    mrtVertexDesc.usage = BufferUsage::Vertex;
    mrtVertexDesc.label = "rhi_mrt_vertices";
    BufferDesc mrtIndexDesc;
    mrtIndexDesc.size = sizeof(mrtIndices);
    mrtIndexDesc.usage = BufferUsage::Index;
    mrtIndexDesc.label = "rhi_mrt_indices";
    auto mrtVertexBuffer =
        device.createBuffer(mrtVertexDesc, mrtPositions.data());
    auto mrtIndexBuffer =
        device.createBuffer(mrtIndexDesc, mrtIndices.data());

    {
        auto invalidEncoder = device.createCommandEncoder();
        auto invalidPass = invalidEncoder->beginRenderPass(target.get());
        bool rejectedDrawWithoutPipeline = false;
        try {
            invalidPass->draw(3);
        } catch (const std::logic_error&) {
            rejectedDrawWithoutPipeline = true;
        }
        require(rejectedDrawWithoutPipeline,
                "draw without a pipeline was accepted");
        invalidPass->end();
    }

    auto encoder = device.createCommandEncoder();
    auto encodedPass = encoder->beginRenderPass(target.get());
    encodedPass->setViewport(0, 0, width, height);
    encodedPass->setPipeline(pipeline.get());
    encodedPass->setVertexBuffer(0, mrtVertexBuffer.get());
    encodedPass->setIndexBuffer(mrtIndexBuffer.get(), IndexFormat::Uint16);
    encodedPass->drawIndexed(3);
    encodedPass->drawIndexed(3);
    encodedPass->end();
    bool rejectedDrawAfterPass = false;
    try {
        encodedPass->draw(3);
    } catch (const std::logic_error&) {
        rejectedDrawAfterPass = true;
    }
    require(rejectedDrawAfterPass, "draw outside an active pass was accepted");
    auto commands = encoder->finish();
    device.submit(*commands);
    for (int frame = 0; frame < 63; ++frame) {
        auto repeatEncoder = device.createCommandEncoder();
        auto repeatPass = repeatEncoder->beginRenderPass(target.get());
        repeatPass->setViewport(0, 0, width, height);
        repeatPass->setPipeline(pipeline.get());
        repeatPass->setVertexBuffer(0, mrtVertexBuffer.get());
        repeatPass->setIndexBuffer(mrtIndexBuffer.get(), IndexFormat::Uint16);
        repeatPass->drawIndexed(3);
        repeatPass->end();
        auto repeatCommands = repeatEncoder->finish();
        device.submit(*repeatCommands);
    }
    const auto g0 = readPixel(device, color0View.get());
    const auto g1 = readPixel(device, color1View.get());
    const auto g2 = readPixel(device, color2View.get());
    expectNear(g0[0], 0.2f, 0.01f, "MRT attachment 0 mismatch");
    expectNear(g0[2], 0.6f, 0.01f, "MRT attachment 0 mismatch");
    expectNear(g1[1], 2.5f, 0.01f, "MRT attachment 1 float mismatch");
    expectNear(g1[2], 3.75f, 0.01f, "MRT attachment 1 float mismatch");
    expectNear(g2[0], 0.8f, 0.01f, "MRT attachment 2 mismatch");
    expectNear(g2[2], 0.1f, 0.01f, "MRT attachment 2 mismatch");

    BindGroupLayoutDesc emptyLayoutDesc;
    auto group0Layout = device.createBindGroupLayout(emptyLayoutDesc);
    auto group1Layout = device.createBindGroupLayout(emptyLayoutDesc);
    auto group2Layout = device.createBindGroupLayout(emptyLayoutDesc);
    BindGroupLayoutDesc consumerGroupDesc;
    consumerGroupDesc.label = "rhi_gbuffer_group3";
    consumerGroupDesc.entries = {
        {0, BindingType::SampledTexture, ShaderStageVisibility::Fragment,
         TextureFormat::RGBA8Unorm},
        {1, BindingType::SampledTexture, ShaderStageVisibility::Fragment,
         TextureFormat::RGBA16Float},
        {2, BindingType::SampledTexture, ShaderStageVisibility::Fragment,
         TextureFormat::RGBA8Unorm},
        {3, BindingType::Sampler, ShaderStageVisibility::Fragment},
        {4, BindingType::UniformBuffer, ShaderStageVisibility::Fragment},
        {5, BindingType::SampledTexture, ShaderStageVisibility::Fragment,
         TextureFormat::Depth32Float, TextureSampleType::Depth},
    };
    auto group3Layout = device.createBindGroupLayout(consumerGroupDesc);
    PipelineLayoutDesc consumerLayoutDesc;
    consumerLayoutDesc.label = "rhi_consumer_layout";
    consumerLayoutDesc.bindGroupLayouts = {
        group0Layout.get(), group1Layout.get(), group2Layout.get(),
        group3Layout.get()};
    auto consumerLayout = device.createPipelineLayout(consumerLayoutDesc);

    const std::array<float, 4> weights = {0.2f, 0.3f, 0.1f, 0.0f};
    BufferDesc weightsDesc;
    weightsDesc.size = sizeof(weights);
    weightsDesc.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
    weightsDesc.label = "rhi_consumer_weights";
    auto weightsBuffer = device.createBuffer(weightsDesc, weights.data());
    auto consumerSampler = device.createSampler();
    BindGroupDesc consumerBindDesc;
    consumerBindDesc.layout = group3Layout.get();
    consumerBindDesc.label = "rhi_gbuffer_bind_group";
    BindGroupEntry g0Entry;
    g0Entry.binding = 0;
    g0Entry.textureView = color0View.get();
    BindGroupEntry g1Entry;
    g1Entry.binding = 1;
    g1Entry.textureView = color1View.get();
    BindGroupEntry g2Entry;
    g2Entry.binding = 2;
    g2Entry.textureView = color2View.get();
    BindGroupEntry samplerEntry;
    samplerEntry.binding = 3;
    samplerEntry.sampler = consumerSampler.get();
    BindGroupEntry weightsEntry;
    weightsEntry.binding = 4;
    weightsEntry.buffer = weightsBuffer.get();
    BindGroupEntry depthSampleEntry;
    depthSampleEntry.binding = 5;
    depthSampleEntry.textureView = depthView.get();
    consumerBindDesc.entries = {
        g0Entry, g1Entry, g2Entry, samplerEntry, weightsEntry,
        depthSampleEntry};
    auto consumerBindGroup = device.createBindGroup(consumerBindDesc);
    BindGroupDesc invalidDepthBind = consumerBindDesc;
    invalidDepthBind.entries[0].textureView = depthView.get();
    expectInvalid([&] { (void)device.createBindGroup(invalidDepthBind); },
                  "depth view was accepted by a color texture binding");
    BindGroupDesc missingBind = consumerBindDesc;
    missingBind.entries.pop_back();
    expectInvalid([&] { (void)device.createBindGroup(missingBind); },
                  "bind group with a missing binding was accepted");
    BindGroupLayoutDesc duplicateBindingLayout = consumerGroupDesc;
    duplicateBindingLayout.entries.push_back(
        duplicateBindingLayout.entries.front());
    expectInvalid(
        [&] { (void)device.createBindGroupLayout(duplicateBindingLayout); },
        "duplicate bind-group layout binding was accepted");

    auto consumerOutput = device.createTexture(textureDesc(
        width, height, TextureFormat::RGBA16Float, attachmentUsage,
        "rhi_consumer_output"));
    auto consumerOutputView =
        makeView(device, consumerOutput.get(), TextureFormat::RGBA16Float,
                 TextureAspect::All, "rhi_consumer_output_view");
    RenderPassDesc consumerPassDesc;
    consumerPassDesc.label = "rhi_consumer_pass";
    consumerPassDesc.colorAttachments.push_back(
        {consumerOutputView.get(), nullptr, LoadOp::Clear, StoreOp::Store,
         {0.0f, 0.0f, 0.0f, 1.0f}});
    auto consumerTarget = device.createRenderTarget(consumerPassDesc);
    GraphicsPipelineDesc consumerPipelineDesc;
    consumerPipelineDesc.label = "rhi_consumer_pipeline";
    consumerPipelineDesc.pipelineLayout = consumerLayout.get();
    consumerPipelineDesc.primitive.cullMode = CullMode::None;
    consumerPipelineDesc.colorTargets = {{TextureFormat::RGBA16Float}};
    consumerPipelineDesc.shader.name = "rhi_consumer_shader";
    consumerPipelineDesc.shader.stages = {
        {FullscreenTriangleVs, ShaderType::Vertex, "main"},
        {GbufferConsumerFs, ShaderType::Fragment, "main"},
    };
    auto consumerPipeline =
        device.createGraphicsPipeline(consumerPipelineDesc);
    auto consumerEncoder = device.createCommandEncoder();
    auto consumerPass =
        consumerEncoder->beginRenderPass(consumerTarget.get());
    consumerPass->setPipeline(consumerPipeline.get());
    bool rejectedMissingGroup = false;
    try {
        consumerPass->draw(3);
    } catch (const std::logic_error&) {
        rejectedMissingGroup = true;
    }
    require(rejectedMissingGroup, "draw without required bind group was accepted");
    consumerPass->setBindGroup(3, consumerBindGroup.get());
    consumerPass->draw(3);
    consumerPass->end();
    auto consumerCommands = consumerEncoder->finish();
    device.submit(*consumerCommands);
    const auto combined = readPixel(device, consumerOutputView.get());
    expectNear(combined[0], 0.495f, 0.015f, "G-buffer consumer red mismatch");
    expectNear(combined[1], 0.86f, 0.015f, "G-buffer consumer green mismatch");
    expectNear(combined[2], 1.255f, 0.02f, "G-buffer consumer blue mismatch");

    struct DisplayModeCheck {
        std::array<float, 4> weights;
        std::array<float, 3> expected;
    };
    const std::array<DisplayModeCheck, 4> displayModes = {{
        {{1.0f, 0.0f, 0.0f, 0.0f}, {0.2f, 0.4f, 0.6f}},
        {{0.0f, 1.0f, 0.0f, 0.0f}, {1.25f, 2.5f, 3.75f}},
        {{0.0f, 0.0f, 1.0f, 0.0f}, {0.8f, 0.3f, 0.1f}},
        {{0.0f, 0.0f, 0.0f, 1.0f}, {0.5f, 0.5f, 0.5f}},
    }};
    for (const auto& mode : displayModes) {
        weightsBuffer->setData(mode.weights.data(), sizeof(mode.weights));
        auto modeEncoder = device.createCommandEncoder();
        auto modePass = modeEncoder->beginRenderPass(consumerTarget.get());
        modePass->setPipeline(consumerPipeline.get());
        modePass->setBindGroup(3, consumerBindGroup.get());
        modePass->draw(3);
        modePass->end();
        auto modeCommands = modeEncoder->finish();
        device.submit(*modeCommands);
        const auto pixel = readPixel(device, consumerOutputView.get());
        expectNear(pixel[0], mode.expected[0], 0.02f,
                   "G-buffer display mode red mismatch");
        expectNear(pixel[1], mode.expected[1], 0.02f,
                   "G-buffer display mode green mismatch");
        expectNear(pixel[2], mode.expected[2], 0.02f,
                   "G-buffer display mode blue mismatch");
    }
    weightsBuffer->setData(weights.data(), sizeof(weights));

    float depthPixel = readPixel(device, depthView.get())[0];
    expectNear(depthPixel, 0.5f, 0.001f, "MRT depth write mismatch");

    auto smallColor = device.createTexture(textureDesc(
        4, 4, TextureFormat::RGBA8Unorm, attachmentUsage, "mrt_small"));
    auto smallView =
        makeView(device, smallColor.get(), TextureFormat::RGBA8Unorm,
                 TextureAspect::All, "mrt_small_view");
    RenderPassDesc mismatch = passDesc;
    mismatch.colorAttachments[2].view = smallView.get();
    expectInvalid([&] { (void)device.createRenderTarget(mismatch); },
                  "mismatched MRT extents were accepted");

    auto sampledOnly = device.createTexture(
        textureDesc(width, height, TextureFormat::RGBA8Unorm,
                    TextureUsage::TextureBinding, "mrt_sampled_only"));
    auto sampledOnlyView =
        makeView(device, sampledOnly.get(), TextureFormat::RGBA8Unorm,
                 TextureAspect::All, "mrt_sampled_only_view");
    RenderPassDesc missingUsage = passDesc;
    missingUsage.colorAttachments[0].view = sampledOnlyView.get();
    expectInvalid([&] { (void)device.createRenderTarget(missingUsage); },
                  "attachment without RenderAttachment usage was accepted");

    RenderPassDesc duplicate = passDesc;
    duplicate.colorAttachments[1].view = color0View.get();
    expectInvalid([&] { (void)device.createRenderTarget(duplicate); },
                  "duplicate MRT texture was accepted");

    TextureResourceDesc msaaDesc =
        textureDesc(width, height, TextureFormat::RGBA8Unorm,
                    TextureUsage::RenderAttachment, "rhi_msaa_color");
    msaaDesc.sampleCount = 4;
    auto msaaColor = device.createTexture(msaaDesc);
    auto msaaColorView =
        makeView(device, msaaColor.get(), TextureFormat::RGBA8Unorm,
                 TextureAspect::All, "rhi_msaa_color_view");
    auto resolvedColor = device.createTexture(textureDesc(
        width, height, TextureFormat::RGBA8Unorm, attachmentUsage,
        "rhi_msaa_resolved"));
    auto resolvedColorView =
        makeView(device, resolvedColor.get(), TextureFormat::RGBA8Unorm,
                 TextureAspect::All, "rhi_msaa_resolved_view");
    RenderPassDesc resolve;
    resolve.label = "rhi_msaa_resolve_pass";
    resolve.colorAttachments = {{
        msaaColorView.get(), resolvedColorView.get(), LoadOp::Clear,
        StoreOp::Store, {0.0f, 0.0f, 0.0f, 1.0f}}};
    auto resolveTarget = device.createRenderTarget(resolve);
    GraphicsPipelineDesc resolvePipelineDesc;
    resolvePipelineDesc.label = "rhi_msaa_resolve_pipeline";
    resolvePipelineDesc.shader.name = "rhi_msaa_resolve_shader";
    resolvePipelineDesc.shader.stages = {
        {FullscreenTriangleVs, ShaderType::Vertex, "main"},
        {ColorProducerFs, ShaderType::Fragment, "main"},
    };
    resolvePipelineDesc.primitive.cullMode = CullMode::None;
    resolvePipelineDesc.colorTargets = {{TextureFormat::RGBA8Unorm}};
    resolvePipelineDesc.sampleCount = 4;
    auto resolvePipeline =
        device.createGraphicsPipeline(resolvePipelineDesc);
    auto resolveEncoder = device.createCommandEncoder();
    auto resolvePass = resolveEncoder->beginRenderPass(resolveTarget.get());
    resolvePass->setPipeline(resolvePipeline.get());
    resolvePass->draw(3);
    resolvePass->end();
    auto resolveCommands = resolveEncoder->finish();
    device.submit(*resolveCommands);
    const auto resolvedPixel = readPixel(device, resolvedColorView.get());
    expectNear(resolvedPixel[0], 0.15f, 0.02f, "MSAA resolve red mismatch");
    expectNear(resolvedPixel[1], 0.25f, 0.02f, "MSAA resolve green mismatch");
    expectNear(resolvedPixel[2], 0.35f, 0.02f, "MSAA resolve blue mismatch");

    RenderPassDesc discard = passDesc;
    discard.colorAttachments[0].storeOp = StoreOp::Discard;
    expectInvalid([&] { (void)device.createRenderTarget(discard); },
                  "unsupported StoreOp::Discard was accepted");

    RenderPassDesc depthOnly;
    depthOnly.depthStencilAttachment = passDesc.depthStencilAttachment;
    auto depthOnlyTarget = device.createRenderTarget(depthOnly);
    GraphicsPipelineDesc depthPipelineDesc;
    depthPipelineDesc.label = "rhi_depth_pipeline";
    depthPipelineDesc.shader.name = "rhi_depth_shader";
    depthPipelineDesc.shader.stages = {
        {FullscreenTriangleVs, ShaderType::Vertex, "main"},
        {DepthOnlyFs, ShaderType::Fragment, "main"},
    };
    depthPipelineDesc.primitive.cullMode = CullMode::None;
    depthPipelineDesc.depthStencil = DepthStencilState{
        TextureFormat::Depth32Float, true, CompareFunction::Less};
    auto depthPipeline = device.createGraphicsPipeline(depthPipelineDesc);
    auto depthEncoder = device.createCommandEncoder();
    auto depthPass = depthEncoder->beginRenderPass(depthOnlyTarget.get());
    depthPass->setPipeline(depthPipeline.get());
    depthPass->draw(3);
    depthPass->end();
    auto depthCommands = depthEncoder->finish();
    device.submit(*depthCommands);
    depthPixel = readPixel(device, depthView.get())[0];
    expectNear(depthPixel, 0.5f, 0.001f, "depth-only pipeline mismatch");

    RenderPassDesc colorOnly;
    colorOnly.colorAttachments.push_back(passDesc.colorAttachments[0]);
    auto colorOnlyTarget = device.createRenderTarget(colorOnly);
    require(colorOnlyTarget->getWidth() == width,
            "color-only target creation failed");

    expectInvalid(
        [&] {
            auto invalidEncoder = device.createCommandEncoder();
            auto invalidPass =
                invalidEncoder->beginRenderPass(colorOnlyTarget.get());
            invalidPass->setPipeline(pipeline.get());
        },
        "three-target pipeline was accepted by a one-target pass");

    GraphicsPipelineDesc colorPipelineDesc;
    colorPipelineDesc.label = "rhi_color_pipeline";
    colorPipelineDesc.shader.name = "rhi_color_shader";
    colorPipelineDesc.shader.stages = {
        {FullscreenTriangleVs, ShaderType::Vertex, "main"},
        {ColorProducerFs, ShaderType::Fragment, "main"},
    };
    colorPipelineDesc.primitive.cullMode = CullMode::None;
    colorPipelineDesc.colorTargets = {{TextureFormat::RGBA8Unorm}};
    auto colorPipeline = device.createGraphicsPipeline(colorPipelineDesc);
    auto colorEncoder = device.createCommandEncoder();
    auto colorPass = colorEncoder->beginRenderPass(colorOnlyTarget.get());
    colorPass->setPipeline(colorPipeline.get());
    colorPass->draw(3);
    colorPass->end();
    auto colorCommands = colorEncoder->finish();
    device.submit(*colorCommands);
    const auto colorPixel = readPixel(device, color0View.get());
    expectNear(colorPixel[0], 0.15f, 0.01f,
               "depth-disabled color pipeline mismatch");

    GraphicsPipelineDesc indexedPipelineDesc = colorPipelineDesc;
    indexedPipelineDesc.label = "rhi_indexed_pipeline";
    indexedPipelineDesc.shader.name = "rhi_indexed_shader";
    indexedPipelineDesc.shader.stages[0].source = IndexedTriangleVs;
    VertexBufferLayout positionLayout;
    positionLayout.arrayStride = sizeof(float) * 2;
    positionLayout.attributes.push_back(
        {VertexFormat::Float32x2, 0, 0});
    indexedPipelineDesc.vertexBuffers = {positionLayout};
    auto indexedPipeline =
        device.createGraphicsPipeline(indexedPipelineDesc);
    const std::array<float, 6> positions = {
        -1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    const std::array<uint16_t, 3> indices = {0, 1, 2};
    BufferDesc vertexDesc;
    vertexDesc.size = sizeof(positions);
    vertexDesc.usage = BufferUsage::Vertex;
    vertexDesc.label = "rhi_indexed_vertices";
    BufferDesc indexDesc;
    indexDesc.size = sizeof(indices);
    indexDesc.usage = BufferUsage::Index;
    indexDesc.label = "rhi_indexed_indices";
    auto vertexBuffer = device.createBuffer(vertexDesc, positions.data());
    auto indexBuffer = device.createBuffer(indexDesc, indices.data());

    auto indexedEncoder = device.createCommandEncoder();
    auto indexedPass =
        indexedEncoder->beginRenderPass(colorOnlyTarget.get());
    indexedPass->setPipeline(indexedPipeline.get());
    bool rejectedMissingBuffers = false;
    try {
        indexedPass->drawIndexed(3);
    } catch (const std::logic_error&) {
        rejectedMissingBuffers = true;
    }
    require(rejectedMissingBuffers,
            "indexed draw without buffers was accepted");
    indexedPass->setVertexBuffer(0, vertexBuffer.get());
    indexedPass->setIndexBuffer(indexBuffer.get(), IndexFormat::Uint16);
    indexedPass->drawIndexed(3);
    indexedPass->end();
    auto indexedCommands = indexedEncoder->finish();
    device.submit(*indexedCommands);
    const auto indexedPixel = readPixel(device, color0View.get());
    expectNear(indexedPixel[1], 0.25f, 0.01f,
               "indexed pipeline draw mismatch");

    GraphicsPipelineDesc invalidDepthDesc = colorPipelineDesc;
    invalidDepthDesc.depthStencil = DepthStencilState{
        TextureFormat::Depth32Float, true, CompareFunction::Less};
    auto invalidDepthPipeline =
        device.createGraphicsPipeline(invalidDepthDesc);
    expectInvalid(
        [&] {
            auto invalidEncoder = device.createCommandEncoder();
            auto invalidPass =
                invalidEncoder->beginRenderPass(colorOnlyTarget.get());
            invalidPass->setPipeline(invalidDepthPipeline.get());
        },
        "depth-enabled pipeline was accepted without a depth attachment");

    require(!isRenderableExtent({0, 12, 1}) &&
                !isRenderableExtent({16, 0, 1}),
            "zero-sized target was considered renderable");
    const std::array<Extent3D, 4> resizeExtents = {
        Extent3D{1, 1, 1}, Extent3D{7, 5, 1}, Extent3D{16, 12, 1},
        Extent3D{31, 17, 1}};
    for (const Extent3D& resizeExtent : resizeExtents) {
    const uint32_t resizedWidth = resizeExtent.width;
    const uint32_t resizedHeight = resizeExtent.height;
    auto resizedG0 = device.createTexture(textureDesc(
        resizedWidth, resizedHeight, TextureFormat::RGBA8Unorm,
        attachmentUsage, "mrt_resized_g0"));
    auto resizedG1 = device.createTexture(textureDesc(
        resizedWidth, resizedHeight, TextureFormat::RGBA16Float,
        attachmentUsage, "mrt_resized_g1"));
    auto resizedG2 = device.createTexture(textureDesc(
        resizedWidth, resizedHeight, TextureFormat::RGBA8Unorm,
        attachmentUsage, "mrt_resized_g2"));
    auto resizedDepth = device.createTexture(textureDesc(
        resizedWidth, resizedHeight, TextureFormat::Depth32Float,
        TextureUsage::RenderAttachment | TextureUsage::TextureBinding |
            TextureUsage::CopySrc,
        "mrt_resized_depth"));
    auto resizedOutput = device.createTexture(textureDesc(
        resizedWidth, resizedHeight, TextureFormat::RGBA16Float,
        attachmentUsage, "mrt_resized_output"));
    auto resizedG0View =
        makeView(device, resizedG0.get(), TextureFormat::RGBA8Unorm,
                 TextureAspect::All, "mrt_resized_g0_view");
    auto resizedG1View =
        makeView(device, resizedG1.get(), TextureFormat::RGBA16Float,
                 TextureAspect::All, "mrt_resized_g1_view");
    auto resizedG2View =
        makeView(device, resizedG2.get(), TextureFormat::RGBA8Unorm,
                 TextureAspect::All, "mrt_resized_g2_view");
    auto resizedDepthView =
        makeView(device, resizedDepth.get(), TextureFormat::Depth32Float,
                 TextureAspect::DepthOnly, "mrt_resized_depth_view");
    auto resizedOutputView =
        makeView(device, resizedOutput.get(), TextureFormat::RGBA16Float,
                 TextureAspect::All, "mrt_resized_output_view");

    RenderPassDesc resizedProducerDesc = passDesc;
    resizedProducerDesc.label = "mrt_resized_producer";
    resizedProducerDesc.colorAttachments[0].view = resizedG0View.get();
    resizedProducerDesc.colorAttachments[1].view = resizedG1View.get();
    resizedProducerDesc.colorAttachments[2].view = resizedG2View.get();
    resizedProducerDesc.depthStencilAttachment->view = resizedDepthView.get();
    auto resizedProducerTarget =
        device.createRenderTarget(resizedProducerDesc);
    RenderPassDesc resizedConsumerDesc = consumerPassDesc;
    resizedConsumerDesc.label = "mrt_resized_consumer";
    resizedConsumerDesc.colorAttachments[0].view = resizedOutputView.get();
    auto resizedConsumerTarget =
        device.createRenderTarget(resizedConsumerDesc);

    BindGroupDesc resizedBindDesc = consumerBindDesc;
    resizedBindDesc.label = "mrt_resized_bind_group";
    resizedBindDesc.entries[0].textureView = resizedG0View.get();
    resizedBindDesc.entries[1].textureView = resizedG1View.get();
    resizedBindDesc.entries[2].textureView = resizedG2View.get();
    resizedBindDesc.entries[5].textureView = resizedDepthView.get();
    auto resizedBindGroup = device.createBindGroup(resizedBindDesc);

    auto resizedEncoder = device.createCommandEncoder();
    auto resizedProducer =
        resizedEncoder->beginRenderPass(resizedProducerTarget.get());
    resizedProducer->setViewport(0, 0, resizedWidth, resizedHeight);
    resizedProducer->setPipeline(pipeline.get());
    resizedProducer->setVertexBuffer(0, mrtVertexBuffer.get());
    resizedProducer->setIndexBuffer(mrtIndexBuffer.get(), IndexFormat::Uint16);
    resizedProducer->drawIndexed(3);
    resizedProducer->end();
    auto resizedConsumer =
        resizedEncoder->beginRenderPass(resizedConsumerTarget.get());
    resizedConsumer->setViewport(0, 0, resizedWidth, resizedHeight);
    resizedConsumer->setPipeline(consumerPipeline.get());
    resizedConsumer->setBindGroup(3, resizedBindGroup.get());
    resizedConsumer->draw(3);
    resizedConsumer->end();
    auto resizedCommands = resizedEncoder->finish();
    device.submit(*resizedCommands);
    require(resizedProducerTarget->getWidth() == resizedWidth &&
                resizedConsumerTarget->getHeight() == resizedHeight,
            "replacement target size mismatch");
    const auto resizedPixel = readPixel(device, resizedOutputView.get());
    expectNear(resizedPixel[0], 0.495f, 0.015f,
               "resized G-buffer consumer red mismatch");
    expectNear(resizedPixel[2], 1.255f, 0.02f,
               "resized G-buffer consumer blue mismatch");
    }

    // Give frame-debuggers a deterministic frame boundary after all offscreen
    // producer/consumer work has completed.
    glfwSwapBuffers(window.getGlfwWindow());
    if (renderDocApi)
        require(renderDocApi->endFrameCapture(nullptr, nullptr) == 1,
                "RenderDoc failed to save the explicit frame capture");
    device.checkError();
    std::cout << "PASS: OpenGL MRT render target, load ops, and validation"
              << std::endl;
    return 0;
}

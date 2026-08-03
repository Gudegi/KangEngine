#include "selection_outline_processor.hpp"
#include "fullscreen_pass.hpp"

namespace KE {

namespace {

static const char* SelectionOutlineVs = R"(
#version 410 core
out vec2 TexCoord;
void main() {
    vec2 position = vec2(
        gl_VertexID == 1 ? 3.0 : -1.0,
        gl_VertexID == 2 ? 3.0 : -1.0);
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = position * 0.5 + 0.5;
}
)";

static const char* SelectionOutlineFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ke_g3_b0;
uniform sampler2D ke_g3_b1;
layout(std140) uniform ke_g3_b3 {
    vec4 uTexelSizeAndRadius;
    vec4 uOutlineColor;
};

void main() {
    vec4 sceneColor = texture(ke_g3_b0, TexCoord);
    float center = texture(ke_g3_b1, TexCoord).r;
    float neighbor = 0.0;

    int radius = int(clamp(ceil(uTexelSizeAndRadius.z), 1.0, 8.0));
    // TODO : Too heavy, use separable pass(ping-phong FBO)
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            vec2 offset = vec2(float(x), float(y)) *
                          uTexelSizeAndRadius.xy;
            vec2 sampleUv = clamp(TexCoord + offset, vec2(0.0), vec2(1.0));
            neighbor = max(neighbor, texture(ke_g3_b1, sampleUv).r);
        }
    }

    float outline = step(0.5, neighbor) * (1.0 - step(0.5, center));
    vec3 color = mix(sceneColor.rgb, uOutlineColor.rgb, outline * uOutlineColor.a);
    FragColor = vec4(color, sceneColor.a);
}
)";

} // namespace

void SelectionOutlineProcessor::init(Backend::GraphicsDevice* device, int width,
                                     int height) {
    _device = device;
    _width = width;
    _height = height;

    for (size_t i = 0; i < 3; ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "selection_outline_empty_group_" + std::to_string(i);
        _groupLayouts[i] = device->createBindGroupLayout(emptyDesc);
    }
    Backend::BindGroupLayoutDesc passLayoutDesc;
    passLayoutDesc.label = "selection_outline_pass_layout";
    passLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::RGBA16Float},
        {1, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::RGBA8Unorm},
        {2, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {3, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _groupLayouts[3] = device->createBindGroupLayout(passLayoutDesc);

    Backend::PipelineLayoutDesc pipelineLayoutDesc;
    pipelineLayoutDesc.label = "selection_outline_pipeline_layout";
    for (const auto& layout : _groupLayouts)
        pipelineLayoutDesc.bindGroupLayouts.push_back(layout.get());
    _pipelineLayout = device->createPipelineLayout(pipelineLayoutDesc);

    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "selection_outline_pipeline";
    pipelineDesc.shader.name = "selection_outline_rhi";
    pipelineDesc.shader.stages = {
        {SelectionOutlineVs, Backend::ShaderType::Vertex, "main"},
        {SelectionOutlineFs, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.pipelineLayout = _pipelineLayout.get();
    _pipeline = device->createGraphicsPipeline(pipelineDesc);

    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "selection_outline_sampler";
    _sampler = device->createSampler(samplerDesc);

    Backend::BufferDesc paramsDesc;
    paramsDesc.size = sizeof(OutlineParams);
    paramsDesc.usage =
        Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
    paramsDesc.label = "selection_outline_params";
    _paramsBuffer = device->createBuffer(paramsDesc);

    // Output format must be 16f for HDR.
    _outputFBO =
        device->createFramebuffer({width, height, false, false, 0,
                                   Backend::FramebufferColorFormat::RGBA16F});
    rebuildOutputTarget();
}

void SelectionOutlineProcessor::rebuildOutputTarget() {
    _outputTarget.reset();
    _outputView.reset();
    if (!_outputFBO || _width <= 0 || _height <= 0)
        return;

    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA16Float;
    viewDesc.label = "selection_outline_output_view";
    _outputView =
        _device->createTextureView(_outputFBO->getColorTexture(), viewDesc);

    Backend::RenderPassDesc passDesc;
    passDesc.label = "selection_outline_composite_pass";
    passDesc.colorAttachments = {{_outputView.get(),
                                  nullptr,
                                  Backend::LoadOp::Clear,
                                  Backend::StoreOp::Store,
                                  {0.0f, 0.0f, 0.0f, 1.0f}}};
    _outputTarget = _device->createRenderTarget(passDesc);
}

void SelectionOutlineProcessor::ensurePassBindings(
    Backend::Texture* sceneColor, Backend::Texture* selectionMask) {
    if (_passBindGroup && _boundScene == sceneColor &&
        _boundMask == selectionMask)
        return;

    Backend::TextureViewDesc sceneDesc;
    sceneDesc.format = Backend::TextureFormat::RGBA16Float;
    sceneDesc.label = "selection_outline_scene_view";
    _sceneView = _device->createTextureView(sceneColor, sceneDesc);
    Backend::TextureViewDesc maskDesc;
    maskDesc.format = Backend::TextureFormat::RGBA8Unorm;
    maskDesc.label = "selection_outline_mask_view";
    _maskView = _device->createTextureView(selectionMask, maskDesc);

    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _groupLayouts[3].get();
    bindDesc.label = "selection_outline_pass_bind_group";
    bindDesc.entries = {
        {0, nullptr, 0, 0, _sceneView.get(), nullptr},
        {1, nullptr, 0, 0, _maskView.get(), nullptr},
        {2, nullptr, 0, 0, nullptr, _sampler.get()},
        {3, _paramsBuffer.get(), 0, sizeof(OutlineParams), nullptr, nullptr},
    };
    _passBindGroup = _device->createBindGroup(bindDesc);
    _boundScene = sceneColor;
    _boundMask = selectionMask;
}

void SelectionOutlineProcessor::renderOutlineCompositePass(
    Backend::Texture* sceneColor, Backend::Texture* selectionMask) {
    if (!sceneColor || !selectionMask || !_outputTarget || _width <= 0 ||
        _height <= 0)
        return;

    ensurePassBindings(sceneColor, selectionMask);
    OutlineParams params;
    params.texelSizeAndRadius =
        glm::vec4(1.0f / static_cast<float>(_width),
                  1.0f / static_cast<float>(_height), _config.radius, 0.0f);
    params.color = _config.color;
    _paramsBuffer->setData(&params, sizeof(params));

    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(_outputTarget.get());
    FullscreenPass::record(*pass, _pipeline.get(),
                           static_cast<uint32_t>(_width),
                           static_cast<uint32_t>(_height),
                           _passBindGroup.get());
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
}

Backend::Texture* SelectionOutlineProcessor::getResult() {
    return _outputFBO->getColorTexture();
}

Backend::Framebuffer* SelectionOutlineProcessor::getOutputFramebuffer() {
    return _outputFBO.get();
}

void SelectionOutlineProcessor::blitToScreen(int width, int height) {
    _outputFBO->blitToScreen(width, height);
}

void SelectionOutlineProcessor::resize(int width, int height) {
    _width = width;
    _height = height;
    _passBindGroup.reset();
    _sceneView.reset();
    _maskView.reset();
    _boundScene = nullptr;
    _boundMask = nullptr;
    if (width <= 0 || height <= 0) {
        _outputTarget.reset();
        _outputView.reset();
        return;
    }
    _outputFBO->resize(width, height);
    rebuildOutputTarget();
}

} // namespace KE

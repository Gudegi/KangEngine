#include "post_processor.hpp"
#include "fullscreen_pass.hpp"
#include <algorithm>

namespace KE {

static const char* RhiToneMapVs = R"(
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

static const char* RhiToneMapFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ke_g3_b0;
layout(std140) uniform ke_g3_b2 {
    vec4 uToneMapParams;
};

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 acesNarkowicz(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Stephen Hill / MJP BakingLab ACES fit.
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
vec3 rrtAndOdtFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 acesFitted(vec3 color) {
    const mat3 acesInputMat = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    const mat3 acesOutputMat = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );

    color = acesInputMat * color;
    color = rrtAndOdtFit(color);
    color = acesOutputMat * color;
    return clamp(color, 0.0, 1.0);
}

void main() {
    vec4 color = texture(ke_g3_b0, TexCoord);
    float gamma = uToneMapParams.x;
    int toneMapMode = int(uToneMapParams.y);
    float exposure = uToneMapParams.z;

    vec3 mapped = color.rgb;
    if (toneMapMode == 1) {
        mapped = mapped * exposure;
        mapped = mapped / (mapped + vec3(1.0));
    } else if (toneMapMode == 2) {
        mapped = vec3(1.0) - exp(-mapped * exposure);
    } else if (toneMapMode == 3) {
        mapped = acesNarkowicz(mapped * exposure);
    } else if (toneMapMode == 4) {
        mapped = acesFitted(mapped * exposure);
    }

    vec3 corrected = pow(max(mapped, vec3(0.0)), vec3(1.0 / gamma));
    FragColor = vec4(corrected, color.a);
}
)";

static const char* RhiBrightExtractFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ke_g3_b0;
layout(std140) uniform ke_g3_b2 {
    vec4 uBrightExtractParams;
};

void main() {
    vec4 color = texture(ke_g3_b0, TexCoord);
    float brightness = max(max(color.r, color.g), color.b);
    vec3 bright = brightness > uBrightExtractParams.x
        ? color.rgb : vec3(0.0);
    FragColor = vec4(bright, color.a);
}
)";

static const char* RhiBlurFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ke_g3_b0;
layout(std140) uniform ke_g3_b2 {
    vec4 uBlurParams;
};

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(ke_g3_b0, 0));
    vec3 result = texture(ke_g3_b0, TexCoord).rgb * 0.227027;
    bool horizontal = uBlurParams.x > 0.5;

    float weights[4] = float[4](0.1945946, 0.1216216, 0.054054, 0.016216);
    for (int i = 0; i < 4; ++i) {
        float offset = float(i + 1);
        vec2 delta = horizontal
            ? vec2(texelSize.x * offset, 0.0)
            : vec2(0.0, texelSize.y * offset);
        result += texture(ke_g3_b0, TexCoord + delta).rgb * weights[i];
        result += texture(ke_g3_b0, TexCoord - delta).rgb * weights[i];
    }

    FragColor = vec4(result, 1.0);
}
)";

static const char* RhiBloomCompositeFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ke_g3_b0;
uniform sampler2D ke_g3_b1;
layout(std140) uniform ke_g3_b3 {
    vec4 uBloomCompositeParams;
};

void main() {
    vec4 scene = texture(ke_g3_b0, TexCoord);
    vec3 bloom = texture(ke_g3_b1, TexCoord).rgb *
                 uBloomCompositeParams.x;
    FragColor = vec4(scene.rgb + bloom, scene.a);
}
)";

void PostProcessor::init(Backend::GraphicsDevice* device, int width,
                         int height) {
    _device = device;
    _width = width;
    _height = height;

    _outputFBO = device->createFramebuffer({width, height, false, false, 0});
    initToneMapRhi();
    initBrightExtractRhi();
    initBlurRhi();
    initBloomCompositeRhi();
}

void PostProcessor::initToneMapRhi() {
    for (size_t i = 0; i < 3; ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "tone_map_empty_group_" + std::to_string(i);
        _toneMapGroupLayouts[i] =
            _device->createBindGroupLayout(emptyDesc);
    }
    Backend::BindGroupLayoutDesc passLayoutDesc;
    passLayoutDesc.label = "tone_map_pass_layout";
    passLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::RGBA16Float},
        {1, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _toneMapGroupLayouts[3] =
        _device->createBindGroupLayout(passLayoutDesc);

    Backend::PipelineLayoutDesc pipelineLayoutDesc;
    pipelineLayoutDesc.label = "tone_map_pipeline_layout";
    for (const auto& layout : _toneMapGroupLayouts)
        pipelineLayoutDesc.bindGroupLayouts.push_back(layout.get());
    _toneMapPipelineLayout =
        _device->createPipelineLayout(pipelineLayoutDesc);

    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "tone_map_pipeline";
    pipelineDesc.shader.name = "tone_map_rhi";
    pipelineDesc.shader.stages = {
        {RhiToneMapVs, Backend::ShaderType::Vertex, "main"},
        {RhiToneMapFs, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA8Unorm}};
    pipelineDesc.pipelineLayout = _toneMapPipelineLayout.get();
    _toneMapPipeline = _device->createGraphicsPipeline(pipelineDesc);

    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "tone_map_sampler";
    _toneMapSampler = _device->createSampler(samplerDesc);

    Backend::BufferDesc paramsDesc;
    paramsDesc.size = sizeof(ToneMapParams);
    paramsDesc.usage = Backend::BufferUsage::Uniform |
                       Backend::BufferUsage::CopyDst;
    paramsDesc.label = "tone_map_params";
    _toneMapParamsBuffer = _device->createBuffer(paramsDesc);
    rebuildToneMapOutputTarget();
}

void PostProcessor::rebuildToneMapOutputTarget() {
    _toneMapOutputTarget.reset();
    _toneMapOutputView.reset();
    if (!_outputFBO || _width <= 0 || _height <= 0)
        return;
    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA8Unorm;
    viewDesc.label = "tone_map_output_view";
    _toneMapOutputView =
        _device->createTextureView(_outputFBO->getColorTexture(), viewDesc);
    Backend::RenderPassDesc passDesc;
    passDesc.label = "tone_map_pass";
    passDesc.colorAttachments = {{_toneMapOutputView.get(), nullptr,
                                  Backend::LoadOp::Clear,
                                  Backend::StoreOp::Store,
                                  {0.0f, 0.0f, 0.0f, 1.0f}}};
    _toneMapOutputTarget = _device->createRenderTarget(passDesc);
}

void PostProcessor::ensureToneMapBindings(Backend::Texture* source) {
    if (_toneMapBindGroup && _boundToneMapSource == source)
        return;
    Backend::TextureViewDesc sourceDesc;
    sourceDesc.format = Backend::TextureFormat::RGBA16Float;
    sourceDesc.label = "tone_map_source_view";
    _toneMapSourceView = _device->createTextureView(source, sourceDesc);
    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _toneMapGroupLayouts[3].get();
    bindDesc.label = "tone_map_pass_bind_group";
    bindDesc.entries = {
        {0, nullptr, 0, 0, _toneMapSourceView.get(), nullptr},
        {1, nullptr, 0, 0, nullptr, _toneMapSampler.get()},
        {2, _toneMapParamsBuffer.get(), 0, sizeof(ToneMapParams), nullptr,
         nullptr},
    };
    _toneMapBindGroup = _device->createBindGroup(bindDesc);
    _boundToneMapSource = source;
}

void PostProcessor::initBrightExtractRhi() {
    for (size_t i = 0; i < 3; ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "bright_extract_empty_group_" + std::to_string(i);
        _brightExtractGroupLayouts[i] =
            _device->createBindGroupLayout(emptyDesc);
    }
    Backend::BindGroupLayoutDesc passLayoutDesc;
    passLayoutDesc.label = "bright_extract_pass_layout";
    passLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::RGBA16Float},
        {1, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _brightExtractGroupLayouts[3] =
        _device->createBindGroupLayout(passLayoutDesc);

    Backend::PipelineLayoutDesc pipelineLayoutDesc;
    pipelineLayoutDesc.label = "bright_extract_pipeline_layout";
    for (const auto& layout : _brightExtractGroupLayouts)
        pipelineLayoutDesc.bindGroupLayouts.push_back(layout.get());
    _brightExtractPipelineLayout =
        _device->createPipelineLayout(pipelineLayoutDesc);

    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "bright_extract_pipeline";
    pipelineDesc.shader.name = "bright_extract_rhi";
    pipelineDesc.shader.stages = {
        {RhiToneMapVs, Backend::ShaderType::Vertex, "main"},
        {RhiBrightExtractFs, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.pipelineLayout = _brightExtractPipelineLayout.get();
    _brightExtractPipeline = _device->createGraphicsPipeline(pipelineDesc);

    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "bright_extract_sampler";
    _brightExtractSampler = _device->createSampler(samplerDesc);

    Backend::BufferDesc paramsDesc;
    paramsDesc.size = sizeof(BrightExtractParams);
    paramsDesc.usage = Backend::BufferUsage::Uniform |
                       Backend::BufferUsage::CopyDst;
    paramsDesc.label = "bright_extract_params";
    _brightExtractParamsBuffer = _device->createBuffer(paramsDesc);
}

void PostProcessor::rebuildBrightExtractOutputTarget() {
    _brightExtractOutputTarget.reset();
    _brightExtractOutputView.reset();
    if (!_bloomPingPongFBO[0] || _bloomWidth <= 0 || _bloomHeight <= 0)
        return;
    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA16Float;
    viewDesc.label = "bright_extract_output_view";
    _brightExtractOutputView = _device->createTextureView(
        _bloomPingPongFBO[0]->getColorTexture(), viewDesc);
    Backend::RenderPassDesc passDesc;
    passDesc.label = "bright_extract_pass";
    passDesc.colorAttachments = {{_brightExtractOutputView.get(), nullptr,
                                  Backend::LoadOp::Clear,
                                  Backend::StoreOp::Store,
                                  {0.0f, 0.0f, 0.0f, 1.0f}}};
    _brightExtractOutputTarget = _device->createRenderTarget(passDesc);
}

void PostProcessor::ensureBrightExtractBindings(Backend::Texture* source) {
    if (_brightExtractBindGroup && _boundBrightExtractSource == source)
        return;
    Backend::TextureViewDesc sourceDesc;
    sourceDesc.format = Backend::TextureFormat::RGBA16Float;
    sourceDesc.label = "bright_extract_source_view";
    _brightExtractSourceView = _device->createTextureView(source, sourceDesc);
    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _brightExtractGroupLayouts[3].get();
    bindDesc.label = "bright_extract_pass_bind_group";
    bindDesc.entries = {
        {0, nullptr, 0, 0, _brightExtractSourceView.get(), nullptr},
        {1, nullptr, 0, 0, nullptr, _brightExtractSampler.get()},
        {2, _brightExtractParamsBuffer.get(), 0, sizeof(BrightExtractParams),
         nullptr, nullptr},
    };
    _brightExtractBindGroup = _device->createBindGroup(bindDesc);
    _boundBrightExtractSource = source;
}

void PostProcessor::initBlurRhi() {
    for (size_t i = 0; i < 3; ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "bloom_blur_empty_group_" + std::to_string(i);
        _blurGroupLayouts[i] = _device->createBindGroupLayout(emptyDesc);
    }
    Backend::BindGroupLayoutDesc passDesc;
    passDesc.label = "bloom_blur_pass_layout";
    passDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::RGBA16Float},
        {1, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _blurGroupLayouts[3] = _device->createBindGroupLayout(passDesc);
    Backend::PipelineLayoutDesc layoutDesc;
    layoutDesc.label = "bloom_blur_pipeline_layout";
    for (const auto& layout : _blurGroupLayouts)
        layoutDesc.bindGroupLayouts.push_back(layout.get());
    _blurPipelineLayout = _device->createPipelineLayout(layoutDesc);
    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "bloom_blur_pipeline";
    pipelineDesc.shader.name = "bloom_blur_rhi";
    pipelineDesc.shader.stages = {
        {RhiToneMapVs, Backend::ShaderType::Vertex, "main"},
        {RhiBlurFs, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.pipelineLayout = _blurPipelineLayout.get();
    _blurPipeline = _device->createGraphicsPipeline(pipelineDesc);
    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "bloom_blur_sampler";
    _blurSampler = _device->createSampler(samplerDesc);
    Backend::BufferDesc bufferDesc;
    bufferDesc.size = sizeof(BlurParams);
    bufferDesc.usage = Backend::BufferUsage::Uniform |
                       Backend::BufferUsage::CopyDst;
    bufferDesc.label = "bloom_blur_params";
    _blurParamsBuffer = _device->createBuffer(bufferDesc);
}

void PostProcessor::rebuildBlurOutputTargets() {
    for (size_t i = 0; i < 2; ++i) {
        _blurOutputTargets[i].reset();
        _blurOutputViews[i].reset();
        if (!_bloomPingPongFBO[i])
            continue;
        Backend::TextureViewDesc viewDesc;
        viewDesc.format = Backend::TextureFormat::RGBA16Float;
        viewDesc.label = "bloom_blur_output_view_" + std::to_string(i);
        _blurOutputViews[i] = _device->createTextureView(
            _bloomPingPongFBO[i]->getColorTexture(), viewDesc);
        Backend::RenderPassDesc passDesc;
        passDesc.label = "bloom_blur_pass_" + std::to_string(i);
        passDesc.colorAttachments = {{_blurOutputViews[i].get(), nullptr,
                                      Backend::LoadOp::Clear,
                                      Backend::StoreOp::Store,
                                      {0.0f, 0.0f, 0.0f, 1.0f}}};
        _blurOutputTargets[i] = _device->createRenderTarget(passDesc);
    }
}

Backend::BindGroup*
PostProcessor::ensureBlurBinding(Backend::Texture* source) {
    size_t slot = source == _bloomPingPongFBO[0]->getColorTexture() ? 0 : 1;
    if (_blurBindGroups[slot] && _boundBlurSources[slot] == source)
        return _blurBindGroups[slot].get();
    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA16Float;
    viewDesc.label = "bloom_blur_source_view_" + std::to_string(slot);
    _blurSourceViews[slot] = _device->createTextureView(source, viewDesc);
    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _blurGroupLayouts[3].get();
    bindDesc.label = "bloom_blur_bind_group_" + std::to_string(slot);
    bindDesc.entries = {
        {0, nullptr, 0, 0, _blurSourceViews[slot].get(), nullptr},
        {1, nullptr, 0, 0, nullptr, _blurSampler.get()},
        {2, _blurParamsBuffer.get(), 0, sizeof(BlurParams), nullptr, nullptr},
    };
    _blurBindGroups[slot] = _device->createBindGroup(bindDesc);
    _boundBlurSources[slot] = source;
    return _blurBindGroups[slot].get();
}

void PostProcessor::initBloomCompositeRhi() {
    for (size_t i = 0; i < 3; ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "bloom_composite_empty_group_" + std::to_string(i);
        _bloomCompositeGroupLayouts[i] =
            _device->createBindGroupLayout(emptyDesc);
    }
    Backend::BindGroupLayoutDesc passDesc;
    passDesc.label = "bloom_composite_pass_layout";
    passDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::RGBA16Float},
        {1, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::RGBA16Float},
        {2, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {3, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _bloomCompositeGroupLayouts[3] =
        _device->createBindGroupLayout(passDesc);
    Backend::PipelineLayoutDesc layoutDesc;
    layoutDesc.label = "bloom_composite_pipeline_layout";
    for (const auto& layout : _bloomCompositeGroupLayouts)
        layoutDesc.bindGroupLayouts.push_back(layout.get());
    _bloomCompositePipelineLayout =
        _device->createPipelineLayout(layoutDesc);
    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "bloom_composite_pipeline";
    pipelineDesc.shader.name = "bloom_composite_rhi";
    pipelineDesc.shader.stages = {
        {RhiToneMapVs, Backend::ShaderType::Vertex, "main"},
        {RhiBloomCompositeFs, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.pipelineLayout = _bloomCompositePipelineLayout.get();
    _bloomCompositePipeline = _device->createGraphicsPipeline(pipelineDesc);
    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "bloom_composite_sampler";
    _bloomCompositeSampler = _device->createSampler(samplerDesc);
    Backend::BufferDesc bufferDesc;
    bufferDesc.size = sizeof(BloomCompositeParams);
    bufferDesc.usage = Backend::BufferUsage::Uniform |
                       Backend::BufferUsage::CopyDst;
    bufferDesc.label = "bloom_composite_params";
    _bloomCompositeParamsBuffer = _device->createBuffer(bufferDesc);
}

void PostProcessor::rebuildBloomCompositeOutputTarget() {
    _bloomCompositeOutputTarget.reset();
    _bloomCompositeOutputView.reset();
    if (!_bloomCompositeFBO)
        return;
    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA16Float;
    viewDesc.label = "bloom_composite_output_view";
    _bloomCompositeOutputView = _device->createTextureView(
        _bloomCompositeFBO->getColorTexture(), viewDesc);
    Backend::RenderPassDesc passDesc;
    passDesc.label = "bloom_composite_pass";
    passDesc.colorAttachments = {{_bloomCompositeOutputView.get(), nullptr,
                                  Backend::LoadOp::Clear,
                                  Backend::StoreOp::Store,
                                  {0.0f, 0.0f, 0.0f, 1.0f}}};
    _bloomCompositeOutputTarget = _device->createRenderTarget(passDesc);
}

void PostProcessor::ensureBloomCompositeBindings(Backend::Texture* scene,
                                                 Backend::Texture* bloom) {
    if (_bloomCompositeBindGroup && _boundCompositeScene == scene &&
        _boundCompositeBloom == bloom)
        return;
    Backend::TextureViewDesc sceneDesc;
    sceneDesc.format = Backend::TextureFormat::RGBA16Float;
    sceneDesc.label = "bloom_composite_scene_view";
    _bloomCompositeSceneView = _device->createTextureView(scene, sceneDesc);
    Backend::TextureViewDesc bloomDesc;
    bloomDesc.format = Backend::TextureFormat::RGBA16Float;
    bloomDesc.label = "bloom_composite_bloom_view";
    _bloomCompositeBloomView = _device->createTextureView(bloom, bloomDesc);
    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _bloomCompositeGroupLayouts[3].get();
    bindDesc.label = "bloom_composite_bind_group";
    bindDesc.entries = {
        {0, nullptr, 0, 0, _bloomCompositeSceneView.get(), nullptr},
        {1, nullptr, 0, 0, _bloomCompositeBloomView.get(), nullptr},
        {2, nullptr, 0, 0, nullptr, _bloomCompositeSampler.get()},
        {3, _bloomCompositeParamsBuffer.get(), 0,
         sizeof(BloomCompositeParams), nullptr, nullptr},
    };
    _bloomCompositeBindGroup = _device->createBindGroup(bindDesc);
    _boundCompositeScene = scene;
    _boundCompositeBloom = bloom;
}

void PostProcessor::process(Backend::Texture* src, float gamma,
                            ToneMapMode toneMapMode, float tonemapExposure,
                            const BloomConfig& bloom) {
    if (!src || _width <= 0 || _height <= 0)
        return;

    Backend::Texture* HDRSource = src;
    if (bloom.enabled) {
        ensureBloomBuffers(bloom);
        renderBrightExtractPass(src, _bloomPingPongFBO[0].get(),
                                bloom.threshold);
        Backend::Texture* blurredBloom =
            renderBloomBlurPass(_bloomPingPongFBO[0]->getColorTexture(), bloom);
        renderBloomCompositePass(src, blurredBloom, _bloomCompositeFBO.get(),
                                 bloom.intensity);
        HDRSource = _bloomCompositeFBO->getColorTexture();
    }

    renderToneMapPass(HDRSource, gamma, toneMapMode, tonemapExposure);
}

void PostProcessor::ensureBloomBuffers(const BloomConfig& bloom) {
    const int downsample = std::max(1, bloom.downsample);
    const int width = std::max(1, _width / downsample);
    const int height = std::max(1, _height / downsample);

    if (!_bloomPingPongFBO[0] || !_bloomPingPongFBO[1] || !_bloomCompositeFBO) {
        Backend::FramebufferDesc bloomDesc;
        bloomDesc.width = width;
        bloomDesc.height = height;
        bloomDesc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
        _bloomPingPongFBO[0] = _device->createFramebuffer(bloomDesc);
        _bloomPingPongFBO[1] = _device->createFramebuffer(bloomDesc);

        Backend::FramebufferDesc compositeDesc;
        compositeDesc.width = _width;
        compositeDesc.height = _height;
        compositeDesc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
        _bloomCompositeFBO = _device->createFramebuffer(compositeDesc);
        _bloomWidth = width;
        _bloomHeight = height;
        _bloomDownsample = downsample;
        rebuildBrightExtractOutputTarget();
        rebuildBlurOutputTargets();
        rebuildBloomCompositeOutputTarget();
        return;
    }

    if (_bloomWidth != width || _bloomHeight != height ||
        _bloomDownsample != downsample) {
        _brightExtractOutputTarget.reset();
        _brightExtractOutputView.reset();
        for (auto& target : _blurOutputTargets)
            target.reset();
        for (auto& view : _blurOutputViews)
            view.reset();
        for (size_t i = 0; i < 2; ++i) {
            _blurBindGroups[i].reset();
            _blurSourceViews[i].reset();
            _boundBlurSources[i] = nullptr;
        }
        _bloomPingPongFBO[0]->resize(width, height);
        _bloomPingPongFBO[1]->resize(width, height);
        _bloomWidth = width;
        _bloomHeight = height;
        _bloomDownsample = downsample;
        rebuildBrightExtractOutputTarget();
        rebuildBlurOutputTargets();
    }

    if (_bloomCompositeFBO->getColorTexture()->getWidth() != _width ||
        _bloomCompositeFBO->getColorTexture()->getHeight() != _height) {
        _bloomCompositeOutputTarget.reset();
        _bloomCompositeOutputView.reset();
        _bloomCompositeFBO->resize(_width, _height);
        rebuildBloomCompositeOutputTarget();
    }
}

void PostProcessor::renderBrightExtractPass(Backend::Texture* src,
                                            Backend::Framebuffer* target,
                                            float threshold) {
    if (!src || target != _bloomPingPongFBO[0].get() ||
        !_brightExtractOutputTarget || _bloomWidth <= 0 || _bloomHeight <= 0)
        return;
    ensureBrightExtractBindings(src);
    BrightExtractParams params;
    params.values.x = threshold;
    _brightExtractParamsBuffer->setData(&params, sizeof(params));

    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(_brightExtractOutputTarget.get());
    FullscreenPass::record(*pass, _brightExtractPipeline.get(),
                           static_cast<uint32_t>(_bloomWidth),
                           static_cast<uint32_t>(_bloomHeight),
                           _brightExtractBindGroup.get());
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
}

Backend::Texture* PostProcessor::renderBloomBlurPass(Backend::Texture* src,
                                                     const BloomConfig& bloom) {
    Backend::Texture* source = src;
    const int iterations = std::max(0, bloom.iterations);
    bool horizontal = true;

    for (int i = 0; i < iterations; ++i) {
        const size_t targetIndex = horizontal ? 1 : 0;
        BlurParams params;
        params.values.x = horizontal ? 1.0f : 0.0f;
        _blurParamsBuffer->setData(&params, sizeof(params));
        auto encoder = _device->createCommandEncoder();
        auto pass =
            encoder->beginRenderPass(_blurOutputTargets[targetIndex].get());
        FullscreenPass::record(*pass, _blurPipeline.get(),
                               static_cast<uint32_t>(_bloomWidth),
                               static_cast<uint32_t>(_bloomHeight),
                               ensureBlurBinding(source));
        pass->end();
        auto commands = encoder->finish();
        _device->submit(*commands);
        source = _bloomPingPongFBO[targetIndex]->getColorTexture();
        horizontal = !horizontal;
    }

    return source;
}

void PostProcessor::renderBloomCompositePass(Backend::Texture* scene,
                                             Backend::Texture* bloom,
                                             Backend::Framebuffer* target,
                                             float intensity) {
    if (!scene || !bloom || target != _bloomCompositeFBO.get() ||
        !_bloomCompositeOutputTarget)
        return;
    ensureBloomCompositeBindings(scene, bloom);
    BloomCompositeParams params;
    params.values.x = intensity;
    _bloomCompositeParamsBuffer->setData(&params, sizeof(params));
    auto encoder = _device->createCommandEncoder();
    auto pass =
        encoder->beginRenderPass(_bloomCompositeOutputTarget.get());
    FullscreenPass::record(*pass, _bloomCompositePipeline.get(),
                           static_cast<uint32_t>(_width),
                           static_cast<uint32_t>(_height),
                           _bloomCompositeBindGroup.get());
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
}

void PostProcessor::renderToneMapPass(Backend::Texture* src, float gamma,
                                      ToneMapMode toneMapMode,
                                      float tonemapExposure) {
    if (!src || !_toneMapOutputTarget || _width <= 0 || _height <= 0)
        return;
    ensureToneMapBindings(src);
    ToneMapParams params;
    params.values =
        glm::vec4(gamma < 0.01f ? 1.0f : gamma,
                  static_cast<float>(toneMapMode), tonemapExposure, 0.0f);
    _toneMapParamsBuffer->setData(&params, sizeof(params));

    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(_toneMapOutputTarget.get());
    FullscreenPass::record(*pass, _toneMapPipeline.get(),
                           static_cast<uint32_t>(_width),
                           static_cast<uint32_t>(_height),
                           _toneMapBindGroup.get());
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
}

Backend::Texture* PostProcessor::getResult() {
    return _outputFBO->getColorTexture();
}

Backend::Framebuffer* PostProcessor::getOutputFramebuffer() {
    return _outputFBO.get();
}

void PostProcessor::blitToScreen(int width, int height) {
    _outputFBO->blitToScreen(width, height);
}

void PostProcessor::resize(int width, int height) {
    _width = width;
    _height = height;
    _toneMapBindGroup.reset();
    _toneMapSourceView.reset();
    _boundToneMapSource = nullptr;
    _brightExtractBindGroup.reset();
    _brightExtractSourceView.reset();
    _boundBrightExtractSource = nullptr;
    for (size_t i = 0; i < 2; ++i) {
        _blurBindGroups[i].reset();
        _blurSourceViews[i].reset();
        _boundBlurSources[i] = nullptr;
    }
    _bloomCompositeBindGroup.reset();
    _bloomCompositeSceneView.reset();
    _bloomCompositeBloomView.reset();
    _boundCompositeScene = nullptr;
    _boundCompositeBloom = nullptr;
    if (width <= 0 || height <= 0) {
        _toneMapOutputTarget.reset();
        _toneMapOutputView.reset();
        _brightExtractOutputTarget.reset();
        _brightExtractOutputView.reset();
        for (auto& target : _blurOutputTargets)
            target.reset();
        for (auto& view : _blurOutputViews)
            view.reset();
        _bloomCompositeOutputTarget.reset();
        _bloomCompositeOutputView.reset();
        return;
    }
    _outputFBO->resize(width, height);
    rebuildToneMapOutputTarget();
    if (_bloomCompositeFBO) {
        _bloomCompositeOutputTarget.reset();
        _bloomCompositeOutputView.reset();
        _bloomCompositeFBO->resize(width, height);
        rebuildBloomCompositeOutputTarget();
    }
    if (_bloomPingPongFBO[0] && _bloomPingPongFBO[1]) {
        const int width = std::max(1, _width / std::max(1, _bloomDownsample));
        const int height = std::max(1, _height / std::max(1, _bloomDownsample));
        _brightExtractOutputTarget.reset();
        _brightExtractOutputView.reset();
        for (auto& target : _blurOutputTargets)
            target.reset();
        for (auto& view : _blurOutputViews)
            view.reset();
        _bloomPingPongFBO[0]->resize(width, height);
        _bloomPingPongFBO[1]->resize(width, height);
        _bloomWidth = width;
        _bloomHeight = height;
        rebuildBrightExtractOutputTarget();
        rebuildBlurOutputTargets();
    }
}

} // namespace KE

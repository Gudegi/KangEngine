#include "native_render_extension.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/graphics/renderer/fullscreen_pass.hpp"
#include "engine/scene/scene_resource_manager.hpp"
#include <algorithm>

namespace KE {
namespace {

constexpr const char* OffscreenCopyVs = R"(
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

constexpr const char* OffscreenCopyFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D ke_g3_b0;
void main() {
    FragColor = texture(ke_g3_b0, TexCoord);
}
)";

} // namespace

NativeRenderExtension::NativeRenderExtension(Backend::GraphicsDevice& device,
                                             Rasterizer& rasterizer,
                                             const RendererSettings& settings)
    : _device(&device), _rasterizer(&rasterizer), _settings(settings) {}

NativeRenderExtension::~NativeRenderExtension() {
    _offscreenCopyBindGroup.reset();
    _offscreenCopyPipeline.reset();
    _offscreenCopySampler.reset();
    _offscreenCopyPipelineLayout.reset();
    for (auto& layout : _offscreenCopyGroupLayouts)
        layout.reset();
    _offscreenOutputTarget.reset();
    _offscreenOutputView.reset();
    _offscreenResolveTarget.reset();
    _offscreenDrawTarget.reset();
    _offscreenClearTarget.reset();
    _offscreenDepthStencilView.reset();
    _offscreenResolveColorView.reset();
    _offscreenMsaaColorView.reset();
    _offscreenMsaaDepthStencil.reset();
    _offscreenIntermediateColor.reset();
    _offscreenMsaaColor.reset();
    _sceneHookFramePipelineLayout.reset();
    _sceneHookDefaultPipelineLayout.reset();
}

void NativeRenderExtension::setViewportSize(int width, int height) {
    _viewportWidth = width;
    _viewportHeight = height;
    _rasterizer->setViewportSize(width, height);
}

std::unique_ptr<Backend::Texture> NativeRenderExtension::createTexture(
    const Scene::TextureSourceResource& source) {
    source.validate();
    if (!source.path.empty())
        return _device->createTexture(source.path, source.flip, source.sampler,
                                      source.srgb);
    const uint8_t* pixels = source.pixels.data();
    std::vector<uint8_t> flipped;
    if (source.flip) {
        flipped.resize(source.pixels.size());
        const size_t rowBytes =
            static_cast<size_t>(source.width) * source.channels;
        for (int row = 0; row < source.height; ++row)
            std::copy_n(pixels + row * rowBytes, rowBytes,
                        flipped.data() + (source.height - 1 - row) * rowBytes);
        pixels = flipped.data();
    }
    Backend::TextureDesc desc{source.width, source.height, source.channels,
                              pixels,       "scene_image", source.srgb};
    return _device->createTexture(desc, source.sampler);
}

std::unique_ptr<Backend::GraphicsPipeline>
NativeRenderExtension::createSceneHookPipeline(
    const SceneHookPipelineDesc& hookDesc) {
    if (!_device || !_rasterizer)
        throw std::runtime_error(
            "scene hook pipeline requires a bound renderer");
    if (hookDesc.shader.stages.empty())
        throw std::invalid_argument(
            "scene hook pipeline requires shader stages");
    for (const Backend::ShaderStage& stage : hookDesc.shader.stages) {
        if (stage.type == Backend::ShaderType::Compute)
            throw std::invalid_argument(
                "compute shaders require a compute pipeline; scene hook "
                "pipelines are graphics-only");
    }

    Backend::PipelineLayout* pipelineLayout = hookDesc.pipelineLayout;
    if (!pipelineLayout && hookDesc.useSceneFrameBindings) {
        if (!_sceneHookFramePipelineLayout) {
            Backend::PipelineLayoutDesc layoutDesc;
            layoutDesc.label = "scene_hook_frame_pipeline_layout";
            layoutDesc.bindGroupLayouts = {
                _rasterizer->sceneFrameBindGroupLayout()};
            _sceneHookFramePipelineLayout =
                _device->createPipelineLayout(layoutDesc);
        }
        pipelineLayout = _sceneHookFramePipelineLayout.get();
    } else if (!pipelineLayout) {
        if (!_sceneHookDefaultPipelineLayout) {
            Backend::PipelineLayoutDesc layoutDesc;
            layoutDesc.label = "scene_hook_default_pipeline_layout";
            _sceneHookDefaultPipelineLayout =
                _device->createPipelineLayout(layoutDesc);
        }
        pipelineLayout = _sceneHookDefaultPipelineLayout.get();
    }

    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = hookDesc.label;
    pipelineDesc.shader = hookDesc.shader;
    pipelineDesc.pipelineLayout = pipelineLayout;
    pipelineDesc.vertexBuffers = hookDesc.vertexBuffers;
    pipelineDesc.primitive.topology = hookDesc.topology;
    pipelineDesc.primitive.cullMode = hookDesc.cullMode;
    pipelineDesc.depthStencil = Backend::DepthStencilState{
        Backend::TextureFormat::Depth24Stencil8, hookDesc.depthWrite,
        hookDesc.depthTest ? hookDesc.depthCompare
                           : Backend::CompareFunction::Always};

    Backend::ColorTargetState colorTarget;
    colorTarget.format = Backend::TextureFormat::RGBA16Float;
    if (hookDesc.blend != SceneHookBlendMode::Opaque) {
        Backend::BlendState blend;
        blend.color.srcFactor = Backend::BlendFactorValue::SrcAlpha;
        blend.alpha.srcFactor = Backend::BlendFactorValue::One;
        if (hookDesc.blend == SceneHookBlendMode::Alpha) {
            blend.color.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
            blend.alpha.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
        } else {
            blend.color.dstFactor = Backend::BlendFactorValue::One;
            blend.alpha.dstFactor = Backend::BlendFactorValue::One;
        }
        colorTarget.blend = blend;
    }
    pipelineDesc.colorTargets = {colorTarget};
    pipelineDesc.sampleCount = 4;
    return _device->createGraphicsPipeline(pipelineDesc);
}

void NativeRenderExtension::ensureOffscreenSceneTarget(
    Backend::Framebuffer* target, int width, int height) {
    Backend::Texture* resolvedColor =
        target ? target->getColorTexture() : nullptr;
    if (!resolvedColor)
        throw std::runtime_error("offscreen scene target has no color texture");
    if (_offscreenDrawTarget && _offscreenFramebuffer == target &&
        _offscreenResolvedColor == resolvedColor && _offscreenWidth == width &&
        _offscreenHeight == height)
        return;

    _offscreenResolveTarget.reset();
    _offscreenDrawTarget.reset();
    _offscreenDepthStencilView.reset();
    _offscreenResolveColorView.reset();
    _offscreenMsaaColorView.reset();
    _offscreenMsaaDepthStencil.reset();
    _offscreenMsaaColor.reset();
    _offscreenIntermediateColor.reset();
    _offscreenOutputTarget.reset();
    _offscreenOutputView.reset();
    _offscreenCopyBindGroup.reset();

    Backend::TextureResourceDesc colorDesc;
    colorDesc.extent = {static_cast<uint32_t>(width),
                        static_cast<uint32_t>(height), 1};
    colorDesc.format = Backend::TextureFormat::RGBA16Float;
    colorDesc.usage = Backend::TextureUsage::RenderAttachment;
    colorDesc.sampleCount = 4;
    colorDesc.label = "offscreen_scene_msaa_color";
    _offscreenMsaaColor = _device->createTexture(colorDesc);

    Backend::Texture* resolveColor = resolvedColor;
    if (resolvedColor->getFormat() != Backend::TextureFormat::RGBA16Float) {
        Backend::TextureResourceDesc intermediateDesc = colorDesc;
        intermediateDesc.sampleCount = 1;
        intermediateDesc.usage = Backend::TextureUsage::RenderAttachment |
                                 Backend::TextureUsage::TextureBinding;
        intermediateDesc.label = "offscreen_scene_intermediate_color";
        _offscreenIntermediateColor = _device->createTexture(intermediateDesc);
        resolveColor = _offscreenIntermediateColor.get();
    }

    Backend::TextureResourceDesc depthDesc;
    depthDesc.extent = colorDesc.extent;
    depthDesc.format = Backend::TextureFormat::Depth24Stencil8;
    depthDesc.usage = Backend::TextureUsage::RenderAttachment;
    depthDesc.sampleCount = 4;
    depthDesc.label = "offscreen_scene_msaa_depth_stencil";
    _offscreenMsaaDepthStencil = _device->createTexture(depthDesc);

    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA16Float;
    viewDesc.label = "offscreen_scene_msaa_color_view";
    _offscreenMsaaColorView =
        _device->createTextureView(_offscreenMsaaColor.get(), viewDesc);
    viewDesc.label = "offscreen_scene_resolve_color_view";
    _offscreenResolveColorView =
        _device->createTextureView(resolveColor, viewDesc);
    viewDesc.format = Backend::TextureFormat::Depth24Stencil8;
    viewDesc.aspect = Backend::TextureAspect::All;
    viewDesc.label = "offscreen_scene_depth_stencil_view";
    _offscreenDepthStencilView =
        _device->createTextureView(_offscreenMsaaDepthStencil.get(), viewDesc);

    Backend::RenderPassDesc passDesc;
    passDesc.label = "offscreen_scene_resolve_pass";
    passDesc.colorAttachments = {{_offscreenMsaaColorView.get(),
                                  _offscreenResolveColorView.get(),
                                  Backend::LoadOp::Load,
                                  Backend::StoreOp::Store,
                                  {}}};
    passDesc.depthStencilAttachment =
        Backend::DepthStencilAttachmentDesc{_offscreenDepthStencilView.get(),
                                            Backend::LoadOp::Load,
                                            Backend::StoreOp::Store,
                                            1.0f,
                                            Backend::LoadOp::Load,
                                            Backend::StoreOp::Store,
                                            0};
    _offscreenResolveTarget = _device->createRenderTarget(passDesc);
    passDesc.label = "offscreen_scene_draw_pass";
    passDesc.colorAttachments[0].resolveTarget = nullptr;
    _offscreenDrawTarget = _device->createRenderTarget(passDesc);
    _offscreenClearTarget.reset();

    if (_offscreenIntermediateColor)
        ensureOffscreenFormatConversion(resolvedColor);

    _offscreenFramebuffer = target;
    _offscreenResolvedColor = resolvedColor;
    _offscreenWidth = width;
    _offscreenHeight = height;
}

void NativeRenderExtension::ensureOffscreenClearTarget(
    const glm::vec4& clearColor) {
    if (_offscreenClearTarget) {
        const auto& current = _offscreenClearTarget->getDesc()
                                  .colorAttachments.front()
                                  .clearValue;
        if (current.r == clearColor.r && current.g == clearColor.g &&
            current.b == clearColor.b && current.a == clearColor.a)
            return;
    }

    Backend::RenderPassDesc passDesc;
    passDesc.label = "offscreen_scene_clear_pass";
    passDesc.colorAttachments = {
        {_offscreenMsaaColorView.get(),
         nullptr,
         Backend::LoadOp::Clear,
         Backend::StoreOp::Store,
         {clearColor.r, clearColor.g, clearColor.b, clearColor.a}}};
    passDesc.depthStencilAttachment =
        Backend::DepthStencilAttachmentDesc{_offscreenDepthStencilView.get(),
                                            Backend::LoadOp::Clear,
                                            Backend::StoreOp::Store,
                                            1.0f,
                                            Backend::LoadOp::Clear,
                                            Backend::StoreOp::Store,
                                            0};
    _offscreenClearTarget = _device->createRenderTarget(passDesc);
}

void NativeRenderExtension::ensureOffscreenFormatConversion(
    Backend::Texture* targetColor) {
    const Backend::TextureFormat outputFormat = targetColor->getFormat();
    if (outputFormat != Backend::TextureFormat::RGBA8Unorm &&
        outputFormat != Backend::TextureFormat::RGBA8UnormSrgb) {
        throw std::runtime_error(
            "unsupported RHI offscreen output color format");
    }

    if (!_offscreenCopyPipelineLayout) {
        for (size_t i = 0; i < 3; ++i) {
            Backend::BindGroupLayoutDesc emptyDesc;
            emptyDesc.label = "offscreen_copy_empty_group_" + std::to_string(i);
            _offscreenCopyGroupLayouts[i] =
                _device->createBindGroupLayout(emptyDesc);
        }
        Backend::BindGroupLayoutDesc copyLayoutDesc;
        copyLayoutDesc.label = "offscreen_copy_group_layout";
        copyLayoutDesc.entries = {
            {0, Backend::BindingType::SampledTexture,
             Backend::ShaderStageVisibility::Fragment,
             Backend::TextureFormat::RGBA16Float},
            {1, Backend::BindingType::Sampler,
             Backend::ShaderStageVisibility::Fragment},
        };
        _offscreenCopyGroupLayouts[3] =
            _device->createBindGroupLayout(copyLayoutDesc);

        Backend::PipelineLayoutDesc layoutDesc;
        layoutDesc.label = "offscreen_copy_pipeline_layout";
        for (const auto& layout : _offscreenCopyGroupLayouts)
            layoutDesc.bindGroupLayouts.push_back(layout.get());
        _offscreenCopyPipelineLayout =
            _device->createPipelineLayout(layoutDesc);

        Backend::SamplerDesc samplerDesc;
        samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
        samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
        samplerDesc.minFilter = Backend::TextureFilter::Linear;
        samplerDesc.magFilter = Backend::TextureFilter::Linear;
        samplerDesc.label = "offscreen_copy_sampler";
        _offscreenCopySampler = _device->createSampler(samplerDesc);
    }

    if (!_offscreenCopyPipeline || _offscreenOutputFormat != outputFormat) {
        Backend::GraphicsPipelineDesc pipelineDesc;
        pipelineDesc.label = "offscreen_copy_pipeline";
        pipelineDesc.shader.name = "offscreen_copy_rhi";
        pipelineDesc.shader.stages = {
            {OffscreenCopyVs, Backend::ShaderType::Vertex, "main"},
            {OffscreenCopyFs, Backend::ShaderType::Fragment, "main"},
        };
        pipelineDesc.primitive.cullMode = Backend::CullMode::None;
        pipelineDesc.colorTargets = {{outputFormat}};
        pipelineDesc.pipelineLayout = _offscreenCopyPipelineLayout.get();
        _offscreenCopyPipeline = _device->createGraphicsPipeline(pipelineDesc);
        _offscreenOutputFormat = outputFormat;
    }

    Backend::TextureViewDesc outputViewDesc;
    outputViewDesc.format = outputFormat;
    outputViewDesc.label = "offscreen_copy_output_view";
    _offscreenOutputView =
        _device->createTextureView(targetColor, outputViewDesc);
    Backend::RenderPassDesc outputPassDesc;
    outputPassDesc.label = "offscreen_copy_pass";
    outputPassDesc.colorAttachments = {{_offscreenOutputView.get(),
                                        nullptr,
                                        Backend::LoadOp::Clear,
                                        Backend::StoreOp::Store,
                                        {0.0f, 0.0f, 0.0f, 1.0f}}};
    _offscreenOutputTarget = _device->createRenderTarget(outputPassDesc);

    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _offscreenCopyGroupLayouts[3].get();
    bindDesc.label = "offscreen_copy_bind_group";
    bindDesc.entries = {
        {0, nullptr, 0, 0, _offscreenResolveColorView.get(), nullptr},
        {1, nullptr, 0, 0, nullptr, _offscreenCopySampler.get()},
    };
    _offscreenCopyBindGroup = _device->createBindGroup(bindDesc);
}

void NativeRenderExtension::recordOffscreenFormatConversion(int width,
                                                            int height) {
    if (!_offscreenIntermediateColor)
        return;
    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(
        _offscreenOutputTarget.get(),
        _device->profilePass("render/offscreen/convert"));
    FullscreenPass::record(
        *pass, _offscreenCopyPipeline.get(), static_cast<uint32_t>(width),
        static_cast<uint32_t>(height), _offscreenCopyBindGroup.get());
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
}

void NativeRenderExtension::renderSceneToFramebuffer(
    const glm::mat4& view, const glm::mat4& proj, Backend::Framebuffer* target,
    int width, int height, bool clear) {
    if (!_rasterizer || !_device || !target || width <= 0 || height <= 0)
        return;

    _rasterizer->setViewportSize(width, height);
    ensureOffscreenSceneTarget(target, width, height);
    if (clear) {
        ensureOffscreenClearTarget(_settings.background.backgroundColor);
        auto encoder = _device->createCommandEncoder();
        auto pass = encoder->beginRenderPass(
            _offscreenClearTarget.get(),
            _device->profilePass("render/offscreen/clear"));
        pass->end();
        auto commands = encoder->finish();
        _device->submit(*commands);
    }
    _rasterizer->render(view, proj, _offscreenDrawTarget.get());
    {
        auto encoder = _device->createCommandEncoder();
        auto pass = encoder->beginRenderPass(
            _offscreenResolveTarget.get(),
            _device->profilePass("render/offscreen/resolve"));
        pass->end();
        auto commands = encoder->finish();
        _device->submit(*commands);
    }
    recordOffscreenFormatConversion(width, height);
    if (_viewportWidth > 0 && _viewportHeight > 0)
        _device->setViewport(0, 0, _viewportWidth, _viewportHeight);
    _rasterizer->setViewportSize(_viewportWidth, _viewportHeight);
}

} // namespace KE

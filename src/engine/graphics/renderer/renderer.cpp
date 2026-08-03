#include "renderer.hpp"

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/camera/camera.hpp"
#include "engine/graphics/renderer/fullscreen_pass.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"

#include <stdexcept>
#include <utility>

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

Renderer::~Renderer() = default;

void Renderer::bind(Backend::GraphicsDevice* device, Rasterizer* rasterizer,
                    PostProcessor* postProcessor,
                    SelectionOutlineProcessor* selectionOutlineProcessor) {
    _device = device;
    _rasterizer = rasterizer;
    _postProcessor = postProcessor;
    _selectionOutlineProcessor = selectionOutlineProcessor;
}

void Renderer::setViewportSize(int width, int height) {
    _viewportWidth = width;
    _viewportHeight = height;
    if (_rasterizer)
        _rasterizer->setViewportSize(width, height);
}

void Renderer::setBackgroundShader(Backend::Shader* shader) {
    _backgroundShader = shader;
    applyBackgroundSettings();
}

void Renderer::applyBackgroundSettings() {
    if (_rasterizer)
        _rasterizer->setBackgroundSettings(_settings.background);
    if (!_backgroundShader)
        return;

    _backgroundShader->use();
    _backgroundShader->setVec4("checkerColor1",
                               _settings.background.checkerColor1);
    _backgroundShader->setVec4("checkerColor2",
                               _settings.background.checkerColor2);
    _backgroundShader->setBool("uShowGrid", _settings.background.showGrid);
    _backgroundShader->setVec4("gridColor", _settings.background.gridColor);
}

void Renderer::setLight(const DirectionalLight& light) {
    if (_rasterizer)
        _rasterizer->setLight(light);
}

const DirectionalLight& Renderer::light() const {
    return _rasterizer->getLight();
}

void Renderer::setPointLights(std::vector<PointLight> lights) {
    if (_rasterizer)
        _rasterizer->setPointLights(std::move(lights));
}

const std::vector<PointLight>& Renderer::pointLights() const {
    return _rasterizer->getPointLights();
}

void Renderer::setSpotLights(std::vector<SpotLight> lights) {
    if (_rasterizer)
        _rasterizer->setSpotLights(std::move(lights));
}

const std::vector<SpotLight>& Renderer::spotLights() const {
    return _rasterizer->getSpotLights();
}

void Renderer::syncSceneLights(Scene::SceneBackend* scene) {
    if (!scene || !scene->getRootPrim())
        return;

    // Renderer light sync only scans this subtree. Define scene lights under
    // /lights so mesh/debug/skeleton prims never join the per-frame search.
    Scene::Prim* lightsRoot = scene->getRootPrim()->getPrimAtPath("/lights");
    if (!lightsRoot)
        return;

    bool hasLightPrim = false;
    bool hasDirectionalLight = false;
    DirectionalLight directionalLight;
    directionalLight.intensity = 0.0f;
    directionalLight.ambient = glm::vec3(0.0f);
    std::vector<PointLight> pointLights;
    std::vector<SpotLight> spotLights;
    pointLights.reserve(MaxPointLights);
    spotLights.reserve(MaxSpotLights);

    lightsRoot->traverse([&](Scene::Prim* prim) {
        if (!prim || prim->getType() != Scene::PrimType::Light)
            return;

        hasLightPrim = true;
        if (!prim->isVisibleInHierarchy())
            return;

        switch (prim->getLightType()) {
        case Scene::LightType::Directional:
            if (!hasDirectionalLight) {
                directionalLight = prim->getDirectionalLight();
                hasDirectionalLight = true;
            }
            break;
        case Scene::LightType::Point:
            if (pointLights.size() < static_cast<size_t>(MaxPointLights))
                pointLights.push_back(prim->getPointLight());
            break;
        case Scene::LightType::Spot:
            if (spotLights.size() < static_cast<size_t>(MaxSpotLights))
                spotLights.push_back(prim->getSpotLight());
            break;
        }
    });

    if (!hasLightPrim)
        return;

    setLight(directionalLight);
    setPointLights(std::move(pointLights));
    setSpotLights(std::move(spotLights));
}

Backend::Framebuffer* Renderer::shadowFbo() {
    return _rasterizer ? _rasterizer->getShadowFbo() : nullptr;
}

RenderHookHandle Renderer::addRenderHook(RenderHookPhase phase,
                                         RenderHookCallback callback) {
    return _rasterizer
               ? _rasterizer->addRenderHook(phase, std::move(callback))
               : InvalidRenderHook;
}

bool Renderer::removeRenderHook(RenderHookHandle handle) {
    return _rasterizer && _rasterizer->removeRenderHook(handle);
}

std::unique_ptr<Backend::GraphicsPipeline>
Renderer::createSceneHookPipeline(const SceneHookPipelineDesc& hookDesc) {
    if (!_device || !_rasterizer)
        throw std::runtime_error(
            "scene hook pipeline requires a bound renderer");
    if (hookDesc.shader.stages.empty())
        throw std::invalid_argument(
            "scene hook pipeline requires shader stages");

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
            blend.color.dstFactor =
                Backend::BlendFactorValue::OneMinusSrcAlpha;
            blend.alpha.dstFactor =
                Backend::BlendFactorValue::OneMinusSrcAlpha;
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

void Renderer::renderSceneToFramebuffer(Camera& camera,
                                        Backend::Framebuffer* target, int width,
                                        int height, bool clear) {
    renderSceneToFramebuffer(camera.getViewMatrix(), camera.getProjMatrix(),
                             target, width, height, clear);
}

void Renderer::ensureOffscreenSceneTarget(Backend::Framebuffer* target,
                                          int width, int height) {
    Backend::Texture* resolvedColor = target ? target->getColorTexture() : nullptr;
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
    passDesc.colorAttachments = {{
        _offscreenMsaaColorView.get(), _offscreenResolveColorView.get(),
        Backend::LoadOp::Load, Backend::StoreOp::Store, {}}};
    passDesc.depthStencilAttachment = Backend::DepthStencilAttachmentDesc{
        _offscreenDepthStencilView.get(), Backend::LoadOp::Load,
        Backend::StoreOp::Store, 1.0f, Backend::LoadOp::Load,
        Backend::StoreOp::Store, 0};
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

void Renderer::ensureOffscreenClearTarget(const glm::vec4& clearColor) {
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
    passDesc.colorAttachments = {{
        _offscreenMsaaColorView.get(), nullptr, Backend::LoadOp::Clear,
        Backend::StoreOp::Store,
        {clearColor.r, clearColor.g, clearColor.b, clearColor.a}}};
    passDesc.depthStencilAttachment = Backend::DepthStencilAttachmentDesc{
        _offscreenDepthStencilView.get(), Backend::LoadOp::Clear,
        Backend::StoreOp::Store, 1.0f, Backend::LoadOp::Clear,
        Backend::StoreOp::Store, 0};
    _offscreenClearTarget = _device->createRenderTarget(passDesc);
}

void Renderer::ensureOffscreenFormatConversion(
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
            emptyDesc.label =
                "offscreen_copy_empty_group_" + std::to_string(i);
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
        _offscreenCopyPipeline =
            _device->createGraphicsPipeline(pipelineDesc);
        _offscreenOutputFormat = outputFormat;
    }

    Backend::TextureViewDesc outputViewDesc;
    outputViewDesc.format = outputFormat;
    outputViewDesc.label = "offscreen_copy_output_view";
    _offscreenOutputView =
        _device->createTextureView(targetColor, outputViewDesc);
    Backend::RenderPassDesc outputPassDesc;
    outputPassDesc.label = "offscreen_copy_pass";
    outputPassDesc.colorAttachments = {{
        _offscreenOutputView.get(), nullptr, Backend::LoadOp::Clear,
        Backend::StoreOp::Store, {0.0f, 0.0f, 0.0f, 1.0f}}};
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

void Renderer::recordOffscreenFormatConversion(int width, int height) {
    if (!_offscreenIntermediateColor)
        return;
    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(_offscreenOutputTarget.get());
    FullscreenPass::record(*pass, _offscreenCopyPipeline.get(),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height),
                           _offscreenCopyBindGroup.get());
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
}

void Renderer::renderSceneToFramebuffer(const glm::mat4& view,
                                        const glm::mat4& proj,
                                        Backend::Framebuffer* target, int width,
                                        int height, bool clear) {
    if (!_rasterizer || !_device || !target || width <= 0 || height <= 0)
        return;

    applyBackgroundSettings();
    _rasterizer->updateFrameData(view, proj);
    _rasterizer->setViewportSize(width, height);
    ensureOffscreenSceneTarget(target, width, height);
    if (clear) {
        ensureOffscreenClearTarget(_settings.background.backgroundColor);
        auto encoder = _device->createCommandEncoder();
        auto pass = encoder->beginRenderPass(_offscreenClearTarget.get());
        pass->end();
        auto commands = encoder->finish();
        _device->submit(*commands);
    }
    _rasterizer->render(view, proj, _offscreenDrawTarget.get());
    {
        auto encoder = _device->createCommandEncoder();
        auto pass = encoder->beginRenderPass(_offscreenResolveTarget.get());
        pass->end();
        auto commands = encoder->finish();
        _device->submit(*commands);
    }
    recordOffscreenFormatConversion(width, height);
    if (_viewportWidth > 0 && _viewportHeight > 0)
        _device->setViewport(0, 0, _viewportWidth, _viewportHeight);
    _rasterizer->setViewportSize(_viewportWidth, _viewportHeight);
}

RenderableHandle Renderer::addRenderable(Material* material, Scene::Prim* prim,
                                         TransformSource transformSource) {
    return _rasterizer
               ? _rasterizer->addRenderable(material, prim, transformSource)
               : InvalidHandle;
}

RenderableHandle Renderer::addSkinnedRenderable(
    Material* material, Scene::Prim* prim,
    const Scene::SkinnedMeshData& skinnedMesh,
    TransformSource transformSource) {
    return _rasterizer ? _rasterizer->addSkinnedRenderable(
                             material, prim, skinnedMesh, transformSource)
                       : InvalidHandle;
}

void Renderer::removePrim(RenderableHandle handle, Scene::Prim* prim) {
    if (_rasterizer)
        _rasterizer->removePrim(handle, prim);
}

void Renderer::removePrim(Scene::Prim* prim) {
    if (_rasterizer)
        _rasterizer->removePrim(prim);
}

void Renderer::updateRenderableTransforms(
    RenderableHandle handle, const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    if (_rasterizer)
        _rasterizer->updateRenderableTransforms(handle, transforms, colors);
}

void Renderer::setRenderableExternalBuffer(RenderableHandle handle,
                                           const ExternalBufferDesc& desc) {
    if (_rasterizer)
        _rasterizer->setRenderableExternalBuffer(handle, desc);
}

std::vector<Sim::GpuArrayView> Renderer::mapRenderableCudaTransformBuffers(
    const std::vector<RenderableHandle>& handles, int count, int deviceId,
    uint64_t streamHandle) {
    if (!_rasterizer)
        throw std::runtime_error("renderer is not initialized");
    return _rasterizer->mapRenderableCudaTransformBuffers(
        handles, count, deviceId, streamHandle);
}

void Renderer::unmapRenderableCudaTransformBuffers(
    const std::vector<RenderableHandle>& handles, int deviceId,
    uint64_t streamHandle) {
    if (!_rasterizer)
        throw std::runtime_error("renderer is not initialized");
    _rasterizer->unmapRenderableCudaTransformBuffers(handles, deviceId,
                                                     streamHandle);
}

bool Renderer::getRenderableInstanceTransform(RenderableHandle handle,
                                              int instanceIndex,
                                              glm::mat4& outTransform) const {
    return _rasterizer && _rasterizer->getRenderableInstanceTransform(
                              handle, instanceIndex, outTransform);
}

bool Renderer::setRenderableInstanceTransform(RenderableHandle handle,
                                              int instanceIndex,
                                              const glm::mat4& transform) {
    return _rasterizer && _rasterizer->setRenderableInstanceTransform(
                              handle, instanceIndex, transform);
}

void Renderer::setRenderableColors(RenderableHandle handle,
                                   const std::vector<glm::vec4>& colors) {
    if (_rasterizer)
        _rasterizer->setRenderableColors(handle, colors);
}

void Renderer::setRenderableDoubleSided(RenderableHandle handle,
                                        bool doubleSided) {
    if (_rasterizer)
        _rasterizer->setRenderableDoubleSided(handle, doubleSided);
}

void Renderer::setRenderableCastsShadow(RenderableHandle handle,
                                        bool castsShadow) {
    if (_rasterizer)
        _rasterizer->setRenderableCastsShadow(handle, castsShadow);
}

void Renderer::setRenderableAlphaMode(RenderableHandle handle, AlphaMode mode,
                                      float cutoff) {
    if (_rasterizer)
        _rasterizer->setRenderableAlphaMode(handle, mode, cutoff);
}

void Renderer::setRenderableTexture(RenderableHandle handle,
                                    Backend::Texture* tex, TextureRole role) {
    setRenderableTexture(handle, tex, textureRoleSlot(role));
}

void Renderer::setRenderableTexture(RenderableHandle handle,
                                    Backend::Texture* tex, int slot) {
    if (_rasterizer)
        _rasterizer->setRenderableTexture(handle, tex, slot);
}

void Renderer::updateRenderableGeometry(RenderableHandle handle,
                                        const std::vector<glm::vec3>& positions,
                                        const std::vector<glm::vec3>& normals) {
    if (_rasterizer)
        _rasterizer->updateRenderableGeometry(handle, positions, normals);
}

void Renderer::updateRenderableSkinningMatrices(
    RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices) {
    if (_rasterizer)
        _rasterizer->updateRenderableSkinningMatrices(handle, boneMatrices);
}

} // namespace KE

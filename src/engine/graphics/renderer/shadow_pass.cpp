#include "engine/graphics/renderer/shadow_pass.hpp"

#include "engine/graphics/renderer/mesh_instancer.hpp"
#include "engine/graphics/renderer/shader_library.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/asset_path.hpp"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <stdexcept>
#include <string>

namespace KE {
namespace {
constexpr const char* DepthOnlyFs = R"(
#version 410 core
void main() {}
)";
} // namespace

size_t ShadowPass::index(bool skinned, bool alphaMask, bool doubleSided) {
    return (skinned ? 4u : 0u) | (alphaMask ? 2u : 0u) |
           (doubleSided ? 1u : 0u);
}

void ShadowPass::createPipeline(bool skinned, bool alphaMask, bool doubleSided,
                                const Backend::GraphicsPipelineDesc& desc) {
    requireInitialized("creating pipelines");
    auto& pipeline = _pipelines[index(skinned, alphaMask, doubleSided)];
    if (pipeline)
        throw std::logic_error("ShadowPass received a duplicate pipeline key");
    pipeline = _device->createGraphicsPipeline(desc);
}

Backend::GraphicsPipeline* ShadowPass::pipelineFor(bool skinned, bool alphaMask,
                                                   bool doubleSided) const {
    auto* pipeline = _pipelines[index(skinned, alphaMask, doubleSided)].get();
    if (!pipeline)
        throw std::logic_error(
            "ShadowPass has no pipeline for the requested key");
    return pipeline;
}

void ShadowPass::initializeResources(Backend::Buffer* shadowBuffer,
                                     ShaderLibrary& shaderLibrary) {
    requireInitialized("initializing resources");
    if (!shadowBuffer)
        throw std::invalid_argument("ShadowPass requires shadow frame data");
    _shaderLibrary = &shaderLibrary;

    Backend::BindGroupLayoutDesc frameDesc;
    frameDesc.label = "shadow_frame_layout";
    frameDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                          Backend::ShaderStageVisibility::Vertex}};
    _groupLayouts[0] = _device->createBindGroupLayout(frameDesc);
    for (size_t i = 1; i < _groupLayouts.size(); ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "shadow_empty_group_" + std::to_string(i);
        _groupLayouts[i] = _device->createBindGroupLayout(emptyDesc);
    }
    Backend::BindGroupLayoutDesc alphaDesc;
    alphaDesc.label = "shadow_alpha_group_layout";
    alphaDesc.entries = {{0, Backend::BindingType::SampledTexture,
                          Backend::ShaderStageVisibility::Fragment},
                         {1, Backend::BindingType::Sampler,
                          Backend::ShaderStageVisibility::Fragment},
                         {2, Backend::BindingType::UniformBuffer,
                          Backend::ShaderStageVisibility::Fragment}};
    _alphaGroupLayout = _device->createBindGroupLayout(alphaDesc);
    Backend::BindGroupLayoutDesc skinDesc;
    skinDesc.label = "shadow_skin_group_layout";
    skinDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                         Backend::ShaderStageVisibility::Vertex}};
    _skinGroupLayout = _device->createBindGroupLayout(skinDesc);
    for (bool skin : {false, true}) {
        for (bool alpha : {false, true}) {
            Backend::PipelineLayoutDesc desc;
            desc.label = "shadow_pipeline_layout";
            desc.bindGroupLayouts = {
                _groupLayouts[0].get(), _groupLayouts[1].get(),
                skin ? _skinGroupLayout.get() : _groupLayouts[2].get(),
                alpha ? _alphaGroupLayout.get() : _groupLayouts[3].get()};
            _pipelineLayouts[(skin ? 2u : 0u) | (alpha ? 1u : 0u)] =
                _device->createPipelineLayout(desc);
        }
    }

    Backend::VertexBufferLayout position;
    position.arrayStride = sizeof(glm::vec3);
    position.attributes = {
        {Backend::VertexFormat::Float32x3, 0, RendererAttribute::Position}};
    Backend::VertexBufferLayout transforms;
    transforms.arrayStride = sizeof(glm::mat4);
    transforms.stepMode = Backend::VertexStepMode::Instance;
    for (uint32_t column = 0; column < 4; ++column)
        transforms.attributes.push_back(
            {Backend::VertexFormat::Float32x4, column * sizeof(glm::vec4),
             static_cast<uint32_t>(RendererAttribute::InstanceTransform0) +
                 column});
    Backend::VertexBufferLayout texCoords;
    texCoords.arrayStride = sizeof(glm::vec2);
    texCoords.attributes = {
        {Backend::VertexFormat::Float32x2, 0, RendererAttribute::TexCoord}};
    Backend::VertexBufferLayout boneIndices;
    boneIndices.arrayStride = sizeof(glm::ivec4);
    boneIndices.attributes = {
        {Backend::VertexFormat::Sint32x4, 0, RendererAttribute::BoneIndices}};
    Backend::VertexBufferLayout boneWeights;
    boneWeights.arrayStride = sizeof(glm::vec4);
    boneWeights.attributes = {
        {Backend::VertexFormat::Float32x4, 0, RendererAttribute::BoneWeights}};
    Backend::VertexBufferLayout empty;
    const std::string staticVs =
        _shaderLibrary->load(KE::getAssetPath("shaders/shadow.vs"));
    const std::string skinVs =
        _shaderLibrary->load(KE::getAssetPath("shaders/skinned_shadow.vs"));
    const std::string alphaFs =
        _shaderLibrary->load(KE::getAssetPath("shaders/shadow.fs"));
    for (bool skin : {false, true}) {
        for (bool alpha : {false, true}) {
            for (bool doubleSided : {false, true}) {
                Backend::GraphicsPipelineDesc desc;
                desc.label = "shadow_rhi_pipeline";
                desc.shader.name = "shadow_rhi";
                desc.shader.stages = {
                    {skin ? skinVs : staticVs, Backend::ShaderType::Vertex,
                     "main"},
                    {alpha ? alphaFs : std::string(DepthOnlyFs),
                     Backend::ShaderType::Fragment, "main"}};
                desc.pipelineLayout =
                    _pipelineLayouts[(skin ? 2u : 0u) | (alpha ? 1u : 0u)]
                        .get();
                desc.vertexBuffers = {position,
                                      transforms,
                                      empty,
                                      alpha ? texCoords : empty,
                                      empty,
                                      empty,
                                      skin ? boneIndices : empty,
                                      skin ? boneWeights : empty};
                desc.primitive.cullMode = doubleSided ? Backend::CullMode::None
                                                      : Backend::CullMode::Back;
                desc.depthStencil = Backend::DepthStencilState{
                    Backend::TextureFormat::Depth32Float,
                    true,
                    Backend::CompareFunction::Less,
                    1,
                    1.0f,
                    0.0f};
                createPipeline(skin, alpha, doubleSided, desc);
            }
        }
    }
    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::Repeat;
    samplerDesc.wrapV = Backend::TextureWrap::Repeat;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "shadow_alpha_sampler";
    _alphaSampler = _device->createSampler(samplerDesc);
    Backend::BindGroupDesc frameGroupDesc;
    frameGroupDesc.layout = _groupLayouts[0].get();
    frameGroupDesc.label = "shadow_frame_bind_group";
    frameGroupDesc.entries = {
        {0, shadowBuffer, 0,
         MaxCascades * sizeof(glm::mat4) + 5 * sizeof(glm::vec4), nullptr,
         nullptr}};
    _frameBindGroup = _device->createBindGroup(frameGroupDesc);
}

ShadowPass::PreparedDraws
ShadowPass::prepare(const std::vector<MeshInstancer*>& casters) {
    PreparedDraws prepared;
    prepared.draws.reserve(casters.size());
    prepared.ownedViews.reserve(casters.size());
    prepared.ownedBindGroups.reserve(casters.size() * 2);
    for (MeshInstancer* inst : casters) {
        PreparedDraw draw;
        draw.instancer = inst;
        draw.skinned = inst->hasSkinning();
        Backend::Texture* alphaTexture = inst->alphaMode() == AlphaMode::Mask
                                             ? inst->alphaMaskTexture()
                                             : nullptr;
        draw.alphaMask = alphaTexture != nullptr;
        draw.pipeline =
            pipelineFor(draw.skinned, draw.alphaMask, inst->isDoubleSided());
        if (alphaTexture) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.format = alphaTexture->getFormat();
            viewDesc.label = "shadow_alpha_view";
            prepared.ownedViews.push_back(
                _device->createTextureView(alphaTexture, viewDesc));
            Backend::BindGroupDesc bindDesc;
            bindDesc.layout = _alphaGroupLayout.get();
            bindDesc.label = "shadow_alpha_bind_group";
            bindDesc.entries = {
                {0, nullptr, 0, 0, prepared.ownedViews.back().get(), nullptr},
                {1, nullptr, 0, 0, nullptr, _alphaSampler.get()},
                {2, inst->alphaParamsBuffer(), 0, sizeof(glm::vec4), nullptr,
                 nullptr}};
            prepared.ownedBindGroups.push_back(
                _device->createBindGroup(bindDesc));
            draw.alphaBindGroup = prepared.ownedBindGroups.back().get();
        }
        if (draw.skinned) {
            Backend::BindGroupDesc bindDesc;
            bindDesc.layout = _skinGroupLayout.get();
            bindDesc.label = "shadow_skin_bind_group";
            bindDesc.entries = {{0, inst->boneMatricesBuffer(), 0,
                                 sizeof(glm::mat4) * Scene::MaxSkinningBones,
                                 nullptr, nullptr}};
            prepared.ownedBindGroups.push_back(
                _device->createBindGroup(bindDesc));
            draw.skinBindGroup = prepared.ownedBindGroups.back().get();
        }
        prepared.draws.push_back(draw);
    }
    return prepared;
}

void ShadowPass::record(Backend::RenderPassEncoder& pass,
                        const PreparedDraws& prepared) const {
    for (const PreparedDraw& draw : prepared.draws) {
        pass.setPipeline(draw.pipeline);
        pass.setBindGroup(0, _frameBindGroup.get());
        if (draw.skinBindGroup)
            pass.setBindGroup(2, draw.skinBindGroup);
        if (draw.alphaBindGroup)
            pass.setBindGroup(3, draw.alphaBindGroup);
        draw.instancer->recordDraw(pass, draw.alphaMask, draw.skinned);
    }
}

void ShadowPass::initializeSampling() {
    requireInitialized("initializing shadow sampling");
    Backend::BindGroupLayoutDesc layoutDesc;
    layoutDesc.label = "shadow_sampling_group_layout";
    for (uint32_t binding = 0; binding < MaxCascades; ++binding)
        layoutDesc.entries.push_back({binding,
                                      Backend::BindingType::SampledTexture,
                                      Backend::ShaderStageVisibility::Fragment,
                                      Backend::TextureFormat::Depth32Float,
                                      Backend::TextureSampleType::Depth});
    layoutDesc.entries.push_back({4, Backend::BindingType::Sampler,
                                  Backend::ShaderStageVisibility::Fragment});
    layoutDesc.entries.push_back({5, Backend::BindingType::UniformBuffer,
                                  Backend::ShaderStageVisibility::Fragment});
    _samplingLayout = _device->createBindGroupLayout(layoutDesc);

    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Nearest;
    samplerDesc.magFilter = Backend::TextureFilter::Nearest;
    samplerDesc.label = "shadow_sampling_sampler";
    _samplingSampler = _device->createSampler(samplerDesc);

    Backend::BufferDesc paramsDesc;
    paramsDesc.size = sizeof(glm::vec4);
    paramsDesc.usage =
        Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
    paramsDesc.label = "shadow_sampling_params";
    _samplingParamsBuffer = _device->createBuffer(paramsDesc);
}

void ShadowPass::initializeDepthTargets(
    Backend::Framebuffer* single,
    const std::array<Backend::Framebuffer*, MaxCascades>& cascades) {
    requireInitialized("initializing shadow depth targets");
    auto createTarget = [&](Backend::Framebuffer* framebuffer,
                            const std::string& label,
                            std::unique_ptr<Backend::TextureView>& view,
                            std::unique_ptr<Backend::RenderTarget>& target) {
        Backend::Texture* depth =
            framebuffer ? framebuffer->getDepthTexture() : nullptr;
        if (!depth)
            throw std::invalid_argument(
                "ShadowPass depth target requires a depth texture");
        Backend::TextureViewDesc viewDesc;
        viewDesc.format = depth->getFormat();
        viewDesc.aspect = Backend::TextureAspect::DepthOnly;
        viewDesc.label = label + "_view";
        view = _device->createTextureView(depth, viewDesc);
        Backend::RenderPassDesc passDesc;
        passDesc.label = label;
        passDesc.depthStencilAttachment = Backend::DepthStencilAttachmentDesc{
            view.get(), Backend::LoadOp::Clear, Backend::StoreOp::Store,
            1.0f,       Backend::LoadOp::Load,  Backend::StoreOp::Store,
            0};
        target = _device->createRenderTarget(passDesc);
    };

    createTarget(single, "shadow_depth_pass", _singleDepthView,
                 _singleDepthTarget);
    for (size_t i = 0; i < cascades.size(); ++i)
        createTarget(cascades[i], "shadow_cascade_pass_" + std::to_string(i),
                     _cascadeDepthViews[i], _cascadeDepthTargets[i]);
}

void ShadowPass::rebuildSampling(
    const std::array<Backend::Texture*, MaxCascades>& textures,
    bool debugCascadeTint) {
    std::array<uintptr_t, MaxCascades> handles{};
    for (size_t i = 0; i < textures.size(); ++i) {
        if (!textures[i])
            throw std::invalid_argument(
                "ShadowPass sampling requires every cascade texture");
        handles[i] = textures[i]->getNativeHandle();
    }
    const glm::vec4 params{debugCascadeTint ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f};
    _samplingParamsBuffer->setData(&params, sizeof(params));
    if (_samplingBindGroup && handles == _samplingHandles)
        return;
    _samplingBindGroup.reset();
    for (size_t i = 0; i < textures.size(); ++i) {
        Backend::TextureViewDesc viewDesc;
        viewDesc.format = Backend::TextureFormat::Depth32Float;
        viewDesc.aspect = Backend::TextureAspect::DepthOnly;
        viewDesc.label = "shadow_sampling_view_" + std::to_string(i);
        _samplingViews[i] = _device->createTextureView(textures[i], viewDesc);
    }
    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _samplingLayout.get();
    bindDesc.label = "shadow_sampling_bind_group";
    for (uint32_t binding = 0; binding < MaxCascades; ++binding)
        bindDesc.entries.push_back(
            {binding, nullptr, 0, 0, _samplingViews[binding].get(), nullptr});
    bindDesc.entries.push_back(
        {4, nullptr, 0, 0, nullptr, _samplingSampler.get()});
    bindDesc.entries.push_back({5, _samplingParamsBuffer.get(), 0,
                                sizeof(glm::vec4), nullptr, nullptr});
    _samplingBindGroup = _device->createBindGroup(bindDesc);
    _samplingHandles = handles;
}

} // namespace KE

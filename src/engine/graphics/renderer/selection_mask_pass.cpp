#include "engine/graphics/renderer/selection_mask_pass.hpp"

#include "engine/graphics/renderer/mesh_instancer.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/asset_path.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <stdexcept>
#include <string>

namespace KE {
namespace {
constexpr const char* OpaqueMaskFs = R"(
#version 410 core
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(1.0); }
)";
} // namespace

size_t SelectionMaskPass::index(bool skinned, bool alphaMask,
                                bool doubleSided) {
    return (skinned ? 4u : 0u) | (alphaMask ? 2u : 0u) |
           (doubleSided ? 1u : 0u);
}

void SelectionMaskPass::initializeResources(Backend::Buffer* cameraBuffer) {
    requireInitialized("initializing resources");
    if (!cameraBuffer)
        throw std::invalid_argument("SelectionMaskPass requires camera data");

    Backend::BindGroupLayoutDesc frameDesc;
    frameDesc.label = "selection_mask_frame_layout";
    frameDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                          Backend::ShaderStageVisibility::Vertex}};
    _groupLayouts[0] = _device->createBindGroupLayout(frameDesc);
    for (size_t i = 1; i < _groupLayouts.size(); ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "selection_mask_empty_group_" + std::to_string(i);
        _groupLayouts[i] = _device->createBindGroupLayout(emptyDesc);
    }
    Backend::BindGroupLayoutDesc alphaDesc;
    alphaDesc.label = "selection_mask_alpha_group_layout";
    alphaDesc.entries = {{0, Backend::BindingType::SampledTexture,
                          Backend::ShaderStageVisibility::Fragment},
                         {1, Backend::BindingType::Sampler,
                          Backend::ShaderStageVisibility::Fragment},
                         {2, Backend::BindingType::UniformBuffer,
                          Backend::ShaderStageVisibility::Fragment}};
    _alphaGroupLayout = _device->createBindGroupLayout(alphaDesc);
    Backend::BindGroupLayoutDesc skinDesc;
    skinDesc.label = "selection_mask_skin_group_layout";
    skinDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                         Backend::ShaderStageVisibility::Vertex}};
    _skinGroupLayout = _device->createBindGroupLayout(skinDesc);

    for (bool skin : {false, true}) {
        for (bool alpha : {false, true}) {
            Backend::PipelineLayoutDesc desc;
            desc.label = "selection_mask_pipeline_layout";
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
    const std::string staticVs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/selection_mask.vs"));
    const std::string skinVs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/skinned_selection_mask.vs"));
    const std::string alphaFs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/selection_mask.fs"));

    for (bool skin : {false, true}) {
        for (bool alpha : {false, true}) {
            for (bool doubleSided : {false, true}) {
                Backend::GraphicsPipelineDesc desc;
                desc.label = "selection_mask_pipeline";
                desc.shader.name = "selection_mask_rhi";
                desc.shader.stages = {
                    {skin ? skinVs : staticVs, Backend::ShaderType::Vertex,
                     "main"},
                    {alpha ? alphaFs : std::string(OpaqueMaskFs),
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
                desc.colorTargets = {{Backend::TextureFormat::RGBA8Unorm}};
                _pipelines[index(skin, alpha, doubleSided)] =
                    _device->createGraphicsPipeline(desc);
            }
        }
    }

    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::Repeat;
    samplerDesc.wrapV = Backend::TextureWrap::Repeat;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "selection_mask_alpha_sampler";
    _alphaSampler = _device->createSampler(samplerDesc);
    Backend::BindGroupDesc frameGroupDesc;
    frameGroupDesc.layout = _groupLayouts[0].get();
    frameGroupDesc.label = "selection_mask_frame_bind_group";
    frameGroupDesc.entries = {
        {0, cameraBuffer, 0, 2 * sizeof(glm::mat4), nullptr, nullptr}};
    _frameBindGroup = _device->createBindGroup(frameGroupDesc);
}

void SelectionMaskPass::ensureTarget(Backend::Framebuffer* framebuffer) {
    Backend::Texture* texture =
        framebuffer ? framebuffer->getColorTexture() : nullptr;
    if (!texture)
        throw std::invalid_argument(
            "SelectionMaskPass target requires a color texture");
    const uintptr_t handle = texture->getNativeHandle();
    if (_outputTarget && _outputHandle == handle &&
        _outputWidth == texture->getWidth() &&
        _outputHeight == texture->getHeight())
        return;
    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA8Unorm;
    viewDesc.label = "selection_mask_output_view";
    _outputView = _device->createTextureView(texture, viewDesc);
    Backend::RenderPassDesc passDesc;
    passDesc.label = "selection_mask_pass";
    passDesc.colorAttachments = {{_outputView.get(),
                                  nullptr,
                                  Backend::LoadOp::Clear,
                                  Backend::StoreOp::Store,
                                  {0.0f, 0.0f, 0.0f, 1.0f}}};
    _outputTarget = _device->createRenderTarget(passDesc);
    _outputHandle = handle;
    _outputWidth = texture->getWidth();
    _outputHeight = texture->getHeight();
}

SelectionMaskPass::PreparedDraw
SelectionMaskPass::prepare(MeshInstancer* instancer, int instanceIndex,
                           Backend::Framebuffer* framebuffer) {
    ensureTarget(framebuffer);
    PreparedDraw draw;
    draw.instancer = instancer;
    draw.instanceIndex = instanceIndex;
    draw.skinned = instancer->hasSkinning();
    Backend::Texture* alphaTexture = instancer->alphaMode() == AlphaMode::Mask
                                         ? instancer->alphaMaskTexture()
                                         : nullptr;
    draw.alphaMask = alphaTexture != nullptr;
    draw.pipeline = _pipelines[index(draw.skinned, draw.alphaMask,
                                     instancer->isDoubleSided())]
                        .get();
    if (alphaTexture) {
        if (!instancer->alphaParamsBuffer())
            throw std::runtime_error(
                "RHI alpha-mask draw is missing its parameter buffer");
        Backend::TextureViewDesc viewDesc;
        viewDesc.format = alphaTexture->getFormat();
        viewDesc.label = "selection_mask_alpha_view";
        draw.alphaView = _device->createTextureView(alphaTexture, viewDesc);
        Backend::BindGroupDesc desc;
        desc.layout = _alphaGroupLayout.get();
        desc.label = "selection_mask_alpha_bind_group";
        desc.entries = {{0, nullptr, 0, 0, draw.alphaView.get(), nullptr},
                        {1, nullptr, 0, 0, nullptr, _alphaSampler.get()},
                        {2, instancer->alphaParamsBuffer(), 0,
                         sizeof(glm::vec4), nullptr, nullptr}};
        draw.alphaBindGroup = _device->createBindGroup(desc);
    }
    if (draw.skinned) {
        if (!instancer->boneMatricesBuffer())
            throw std::runtime_error(
                "RHI skinned selection mask is missing its bone buffer");
        Backend::BindGroupDesc desc;
        desc.layout = _skinGroupLayout.get();
        desc.label = "selection_mask_skin_bind_group";
        desc.entries = {{0, instancer->boneMatricesBuffer(), 0,
                         sizeof(glm::mat4) * Scene::MaxSkinningBones, nullptr,
                         nullptr}};
        draw.skinBindGroup = _device->createBindGroup(desc);
    }
    return draw;
}

void SelectionMaskPass::record(Backend::RenderPassEncoder& pass,
                               const PreparedDraw& draw) const {
    pass.setPipeline(draw.pipeline);
    pass.setBindGroup(0, _frameBindGroup.get());
    if (draw.skinBindGroup)
        pass.setBindGroup(2, draw.skinBindGroup.get());
    if (draw.alphaBindGroup)
        pass.setBindGroup(3, draw.alphaBindGroup.get());
    draw.instancer->recordInstanceMask(pass, draw.instanceIndex, draw.alphaMask,
                                       draw.skinned);
}

} // namespace KE

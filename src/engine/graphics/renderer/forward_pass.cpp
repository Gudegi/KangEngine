#include "engine/graphics/renderer/forward_pass.hpp"

#include "engine/graphics/material/material.hpp"
#include "engine/graphics/renderer/mesh_instancer.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/graphics/renderer/shader_library.hpp"
#include "utils/asset_path.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace KE {

Backend::BindGroupLayout*
ForwardPass::createBindGroupLayout(const Backend::BindGroupLayoutDesc& desc) {
    requireInitialized("creating layouts");
    _bindGroupLayouts.push_back(_device->createBindGroupLayout(desc));
    return _bindGroupLayouts.back().get();
}

Backend::PipelineLayout*
ForwardPass::createPipelineLayout(const Backend::PipelineLayoutDesc& desc) {
    requireInitialized("creating layouts");
    _pipelineLayouts.push_back(_device->createPipelineLayout(desc));
    return _pipelineLayouts.back().get();
}

Backend::BindGroup*
ForwardPass::createBindGroup(const Backend::BindGroupDesc& desc) {
    requireInitialized("creating resources");
    _bindGroups.push_back(_device->createBindGroup(desc));
    return _bindGroups.back().get();
}

Backend::Buffer* ForwardPass::createBuffer(const Backend::BufferDesc& desc) {
    requireInitialized("creating resources");
    _buffers.push_back(_device->createBuffer(desc));
    return _buffers.back().get();
}

Backend::Sampler* ForwardPass::createSampler(const Backend::SamplerDesc& desc) {
    requireInitialized("creating resources");
    _samplers.push_back(_device->createSampler(desc));
    return _samplers.back().get();
}

Backend::Texture*
ForwardPass::createTexture(const Backend::TextureResourceDesc& desc,
                           const Backend::TextureInitialData* initialData) {
    requireInitialized("creating resources");
    _textures.push_back(_device->createTexture(desc, initialData));
    return _textures.back().get();
}

void ForwardPass::createPipeline(RasterPipelineKey key,
                                 const Backend::GraphicsPipelineDesc& desc) {
    requireInitialized("creating pipelines");
    key.pass = RasterPassSignature::fromPipelineDesc(desc);
    key.shaderGeneration = _shaderGeneration;
    if (!_passSignature)
        _passSignature = key.pass;
    else if (!(*_passSignature == key.pass))
        throw std::logic_error(
            "ForwardPass pipeline variants must share one pass signature");
    _pipelines.getOrCreate(
        key, [&] { return _device->createGraphicsPipeline(desc); });
}

Backend::GraphicsPipeline* ForwardPass::pipelineFor(const MeshInstancer& inst,
                                                    bool transparent) const {
    RasterPipelineKey key;
    if (!_passSignature)
        throw std::logic_error(
            "ForwardPass has no initialized render-pass signature");
    key.pass = *_passSignature;
    key.shaderGeneration = _shaderGeneration;
    key.skinned = inst.hasSkinning();
    key.transparent = transparent;
    key.doubleSided = inst.isDoubleSided();

    const Material* material = inst.material();
    if (material->shadingModel() == MaterialShadingModel::VertexColor) {
        switch (material->vertexColorStyle()) {
        case VertexColorStyle::Textured:
            key.family = RasterPipelineFamily::TexturedVertexColor;
            break;
        case VertexColorStyle::Checkerboard:
            key.family = RasterPipelineFamily::Checkerboard;
            // Checkerboard is a ground-only family and has no skin input.
            key.skinned = false;
            break;
        case VertexColorStyle::DebugChecker:
            key.family = RasterPipelineFamily::DebugChecker;
            break;
        default:
            key.family = RasterPipelineFamily::VertexColor;
            break;
        }
    } else if (material->shadingModel() == MaterialShadingModel::Phong) {
        key.family = RasterPipelineFamily::Phong;
    } else if (material->shadingModel() == MaterialShadingModel::PBR) {
        key.family = RasterPipelineFamily::Pbr;
    } else {
        key.family = RasterPipelineFamily::VertexColor;
    }

    return _pipelines.get(key);
}

void ForwardPass::configureMaterialBindings(
    Backend::BindGroupLayout* texturedVertexColorLayout,
    Backend::BindGroupLayout* phongLayout, Backend::BindGroupLayout* pbrLayout,
    Backend::Sampler* sampler, Backend::Texture* whiteTexture,
    Backend::Texture* normalTexture) {
    _texturedVertexColorLayout = texturedVertexColorLayout;
    _phongLayout = phongLayout;
    _pbrLayout = pbrLayout;
    _materialSampler = sampler;
    _whiteTexture = whiteTexture;
    _normalTexture = normalTexture;
}

Backend::BindGroup*
ForwardPass::updatePhongMaterial(PhongMaterial& material,
                                 const MeshInstancer& inst) {
    struct alignas(16) Params {
        glm::vec4 ambientShininess;
        glm::vec4 diffuseAlphaCutoff;
        glm::vec4 specularAlphaMode;
        glm::vec4 textureFlags;
    };
    auto [it, inserted] = _phongResources.try_emplace(&material);
    PhongResources& resources = it->second;
    if (inserted) {
        Backend::BufferDesc desc;
        desc.size = sizeof(Params);
        desc.usage =
            Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
        desc.label = "phong_material_params";
        resources.params = _device->createBuffer(desc);
    }
    const Params params{
        glm::vec4(material.ambient, material.shininess),
        glm::vec4(material.diffuse, inst.alphaCutoff()),
        glm::vec4(material.specular,
                  static_cast<float>(static_cast<int>(inst.alphaMode()))),
        glm::vec4(material.diffuseMap ? 1.0f : 0.0f,
                  material.specularMap ? 1.0f : 0.0f,
                  material.alphaMap ? 1.0f : 0.0f,
                  material.normalMap ? 1.0f : 0.0f)};
    resources.params->setData(&params, sizeof(params));

    std::array<Backend::Texture*, 4> textures{
        material.diffuseMap ? material.diffuseMap : _whiteTexture,
        material.specularMap ? material.specularMap : _whiteTexture,
        material.alphaMap ? material.alphaMap : _whiteTexture,
        material.normalMap ? material.normalMap : _normalTexture};
    std::array<uintptr_t, 4> handles{};
    for (size_t i = 0; i < textures.size(); ++i)
        handles[i] = textures[i]->getNativeHandle();
    if (!resources.bindGroup || handles != resources.textureHandles) {
        resources.bindGroup.reset();
        for (size_t i = 0; i < textures.size(); ++i) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.label = "phong_material_view_" + std::to_string(i);
            resources.views[i] =
                _device->createTextureView(textures[i], viewDesc);
        }
        Backend::BindGroupDesc desc;
        desc.layout = _phongLayout;
        desc.label = "phong_material_bind_group";
        desc.entries = {
            {0, resources.params.get(), 0, sizeof(Params), nullptr, nullptr},
            {1, nullptr, 0, 0, resources.views[0].get(), nullptr},
            {2, nullptr, 0, 0, resources.views[1].get(), nullptr},
            {3, nullptr, 0, 0, resources.views[2].get(), nullptr},
            {4, nullptr, 0, 0, resources.views[3].get(), nullptr},
            {5, nullptr, 0, 0, nullptr, _materialSampler},
        };
        resources.bindGroup = _device->createBindGroup(desc);
        resources.textureHandles = handles;
    }
    return resources.bindGroup.get();
}

Backend::BindGroup* ForwardPass::updatePbrMaterial(PBRMaterial& material,
                                                   const MeshInstancer& inst) {
    struct alignas(16) Params {
        glm::vec4 baseColor;
        glm::vec4 factors;
        glm::vec4 emissiveAlpha;
        glm::vec4 textureFlags0;
        glm::vec4 textureFlags1;
    };
    auto [it, inserted] = _pbrResources.try_emplace(&material);
    PbrResources& resources = it->second;
    if (inserted) {
        Backend::BufferDesc desc;
        desc.size = sizeof(Params);
        desc.usage =
            Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
        desc.label = "pbr_material_params";
        resources.params = _device->createBuffer(desc);
    }
    const Params params{
        material.baseColor,
        glm::vec4(material.metallic, material.roughness,
                  material.emissiveStrength, inst.alphaCutoff()),
        glm::vec4(material.emissiveColor,
                  static_cast<float>(static_cast<int>(inst.alphaMode()))),
        glm::vec4(material.baseColorTexture ? 1.0f : 0.0f,
                  material.normalTexture ? 1.0f : 0.0f,
                  material.metallicRoughnessTexture ? 1.0f : 0.0f,
                  material.metallicTexture ? 1.0f : 0.0f),
        glm::vec4(material.roughnessTexture ? 1.0f : 0.0f,
                  material.aoTexture ? 1.0f : 0.0f,
                  material.ormTexture ? 1.0f : 0.0f,
                  material.emissiveTexture ? 1.0f : 0.0f)};
    resources.params->setData(&params, sizeof(params));

    std::array<Backend::Texture*, 8> textures{
        material.baseColorTexture ? material.baseColorTexture : _whiteTexture,
        material.normalTexture ? material.normalTexture : _normalTexture,
        material.metallicRoughnessTexture ? material.metallicRoughnessTexture
                                          : _whiteTexture,
        material.metallicTexture ? material.metallicTexture : _whiteTexture,
        material.roughnessTexture ? material.roughnessTexture : _whiteTexture,
        material.aoTexture ? material.aoTexture : _whiteTexture,
        material.ormTexture ? material.ormTexture : _whiteTexture,
        material.emissiveTexture ? material.emissiveTexture : _whiteTexture};
    std::array<uintptr_t, 8> handles{};
    for (size_t i = 0; i < textures.size(); ++i)
        handles[i] = textures[i]->getNativeHandle();
    if (!resources.bindGroup || handles != resources.textureHandles) {
        resources.bindGroup.reset();
        for (size_t i = 0; i < textures.size(); ++i) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.label = "pbr_material_view_" + std::to_string(i);
            resources.views[i] =
                _device->createTextureView(textures[i], viewDesc);
        }
        Backend::BindGroupDesc desc;
        desc.layout = _pbrLayout;
        desc.label = "pbr_material_bind_group";
        desc.entries.push_back(
            {0, resources.params.get(), 0, sizeof(Params), nullptr, nullptr});
        for (uint32_t binding = 1; binding <= 8; ++binding)
            desc.entries.push_back({binding, nullptr, 0, 0,
                                    resources.views[binding - 1].get(),
                                    nullptr});
        desc.entries.push_back({9, nullptr, 0, 0, nullptr, _materialSampler});
        resources.bindGroup = _device->createBindGroup(desc);
        resources.textureHandles = handles;
    }
    return resources.bindGroup.get();
}

Backend::BindGroup*
ForwardPass::updateTexturedVertexColorMaterial(const MeshInstancer& inst) {
    auto [it, inserted] = _texturedVertexColorResources.try_emplace(&inst);
    TexturedVertexColorResources& resources = it->second;
    if (inserted) {
        Backend::BufferDesc desc;
        desc.size = sizeof(glm::vec4);
        desc.usage =
            Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
        desc.label = "textured_vertex_color_params";
        resources.params = _device->createBuffer(desc);
    }

    Backend::Texture* baseColor =
        inst.textureAtSlot(RendererTextureSlot::BaseColor);
    Backend::Texture* normal = inst.textureAtSlot(RendererTextureSlot::Normal);
    const glm::vec4 params{
        normal && inst.hasTangents() ? 1.0f : 0.0f,
        static_cast<float>(static_cast<int>(inst.alphaMode())),
        inst.alphaCutoff(), 0.0f};
    resources.params->setData(&params, sizeof(params));

    std::array<Backend::Texture*, 2> textures{baseColor ? baseColor
                                                        : _whiteTexture,
                                              normal ? normal : _normalTexture};
    const std::array<const Backend::Texture*, 2> identities{textures[0],
                                                            textures[1]};
    if (!resources.bindGroup || resources.textures != identities) {
        resources.bindGroup.reset();
        for (size_t i = 0; i < textures.size(); ++i) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.label = "textured_vertex_color_view_" + std::to_string(i);
            resources.views[i] =
                _device->createTextureView(textures[i], viewDesc);
        }
        Backend::BindGroupDesc desc;
        desc.layout = _texturedVertexColorLayout;
        desc.label = "textured_vertex_color_bind_group";
        desc.entries = {
            {0, nullptr, 0, 0, resources.views[0].get(), nullptr},
            {1, nullptr, 0, 0, resources.views[1].get(), nullptr},
            {2, nullptr, 0, 0, nullptr, _materialSampler},
            {3, resources.params.get(), 0, sizeof(glm::vec4), nullptr, nullptr},
        };
        resources.bindGroup = _device->createBindGroup(desc);
        resources.textures = identities;
    }
    return resources.bindGroup.get();
}

ForwardPass::PreparedDraws
ForwardPass::prepare(const std::vector<MeshInstancer*>& drawables,
                     bool transparent) {
    PreparedDraws prepared;
    prepared.draws.reserve(drawables.size());
    prepared.ownedBindGroups.reserve(drawables.size());
    for (MeshInstancer* inst : drawables) {
        Material* material = inst->material();
        const bool vertexColor =
            material->shadingModel() == MaterialShadingModel::VertexColor;
        const bool textured = vertexColor && material->vertexColorStyle() ==
                                                 VertexColorStyle::Textured;
        const bool checkerboard =
            vertexColor &&
            material->vertexColorStyle() == VertexColorStyle::Checkerboard;
        const bool debugChecker =
            vertexColor &&
            material->vertexColorStyle() == VertexColorStyle::DebugChecker;

        PreparedDraw draw;
        draw.instancer = inst;
        draw.pipeline = pipelineFor(*inst, transparent);

        if (inst->hasSkinning()) {
            Backend::BindGroupDesc skinDesc;
            skinDesc.layout = _skinGroupLayout;
            skinDesc.label = transparent ? "transparent_skin_bind_group"
                                         : "forward_skin_bind_group";
            skinDesc.entries = {{0, inst->boneMatricesBuffer(), 0,
                                 inst->boneMatricesBuffer()->getSize(), nullptr,
                                 nullptr}};
            prepared.ownedBindGroups.push_back(
                _device->createBindGroup(skinDesc));
            draw.skinBindGroup = prepared.ownedBindGroups.back().get();

            if (textured) {
                draw.kind = DrawKind::SkinnedMaterial;
                draw.materialBindGroup =
                    updateTexturedVertexColorMaterial(*inst);
                draw.normalMapped =
                    inst->textureAtSlot(RendererTextureSlot::Normal) != nullptr;
            } else if (debugChecker) {
                draw.kind = DrawKind::SkinnedMaterial;
            } else if (vertexColor) {
                draw.kind = DrawKind::SkinnedForward;
            } else if (material->shadingModel() ==
                       MaterialShadingModel::Phong) {
                auto* phong = static_cast<PhongMaterial*>(material);
                draw.kind = DrawKind::SkinnedMaterial;
                draw.materialBindGroup = updatePhongMaterial(*phong, *inst);
                draw.normalMapped = phong->normalMap != nullptr;
            } else {
                auto* pbr = static_cast<PBRMaterial*>(material);
                draw.kind = DrawKind::SkinnedMaterial;
                draw.materialBindGroup = updatePbrMaterial(*pbr, *inst);
                draw.normalMapped = pbr->normalTexture != nullptr;
            }
        } else if (debugChecker) {
            draw.kind = DrawKind::Material;
        } else if (checkerboard) {
            draw.kind = DrawKind::Material;
            draw.materialBindGroup = _checkerboardBindGroup;
        } else if (textured) {
            draw.kind = DrawKind::Material;
            draw.materialBindGroup = updateTexturedVertexColorMaterial(*inst);
            draw.normalMapped =
                inst->textureAtSlot(RendererTextureSlot::Normal) != nullptr;
        } else if (material->shadingModel() == MaterialShadingModel::Phong) {
            auto* phong = static_cast<PhongMaterial*>(material);
            draw.kind = DrawKind::Material;
            draw.materialBindGroup = updatePhongMaterial(*phong, *inst);
            draw.normalMapped = phong->normalMap != nullptr;
        } else if (material->shadingModel() == MaterialShadingModel::PBR) {
            auto* pbr = static_cast<PBRMaterial*>(material);
            draw.kind = DrawKind::Material;
            draw.materialBindGroup = updatePbrMaterial(*pbr, *inst);
            draw.normalMapped = pbr->normalTexture != nullptr;
        }
        prepared.draws.push_back(draw);
    }
    return prepared;
}

void ForwardPass::record(Backend::RenderPassEncoder& pass,
                         const PreparedDraws& prepared,
                         Backend::BindGroup* shadowBindGroup) const {
    for (const PreparedDraw& draw : prepared.draws) {
        pass.setPipeline(draw.pipeline);
        pass.setBindGroup(0, _frameBindGroup);
        pass.setBindGroup(1, shadowBindGroup);
        if (draw.skinBindGroup)
            pass.setBindGroup(2, draw.skinBindGroup);
        if (draw.materialBindGroup)
            pass.setBindGroup(3, draw.materialBindGroup);

        switch (draw.kind) {
        case DrawKind::Forward:
            draw.instancer->recordForwardDraw(pass);
            break;
        case DrawKind::Material:
            draw.instancer->recordMaterialDraw(pass, true, draw.normalMapped);
            break;
        case DrawKind::SkinnedForward:
            draw.instancer->recordSkinnedForwardDraw(pass);
            break;
        case DrawKind::SkinnedMaterial:
            draw.instancer->recordSkinnedMaterialDraw(pass, draw.normalMapped);
            break;
        }
    }
}

void ForwardPass::initializeResources(
    Backend::Buffer* cameraBuffer, Backend::Buffer* lightBuffer,
    Backend::Buffer* shadowBuffer,
    Backend::BindGroupLayout* shadowSamplingLayout,
    ShaderLibrary& shaderLibrary) {
    _shaderLibrary = &shaderLibrary;
    _shaderGeneration = shaderLibrary.generation();
    // Group 0 owns frame data. Group 1 reuses the shadow-sampling layout;
    // groups 2/3 stay reserved for skinning and material resources.
    Backend::BindGroupLayoutDesc frameDesc;
    frameDesc.label = "forward_frame_group_layout";
    frameDesc.entries = {
        {0, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Vertex},
        {1, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _groupLayouts[0] = createBindGroupLayout(frameDesc);
    for (size_t i = 1; i < _groupLayouts.size(); ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "forward_reserved_group_" + std::to_string(i + 1);
        _groupLayouts[i] = createBindGroupLayout(emptyDesc);
    }

    Backend::PipelineLayoutDesc layoutDesc;
    layoutDesc.label = "forward_vertex_color_pipeline_layout";
    layoutDesc.bindGroupLayouts = {_groupLayouts[0], shadowSamplingLayout,
                                   _groupLayouts[1], _groupLayouts[2]};
    _forwardPipelineLayout = createPipelineLayout(layoutDesc);

    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _groupLayouts[0];
    bindDesc.label = "forward_frame_bind_group";
    bindDesc.entries = {
        {0, cameraBuffer, 0, cameraBuffer->getSize(), nullptr, nullptr},
        {1, lightBuffer, 0, lightBuffer->getSize(), nullptr, nullptr},
        {2, shadowBuffer, 0, shadowBuffer->getSize(), nullptr, nullptr},
    };
    _frameBindGroup = createBindGroup(bindDesc);

    Backend::VertexBufferLayout position;
    position.arrayStride = sizeof(glm::vec3);
    position.attributes = {
        {Backend::VertexFormat::Float32x3, 0, RendererAttribute::Position}};
    Backend::VertexBufferLayout transform;
    transform.arrayStride = sizeof(glm::mat4);
    transform.stepMode = Backend::VertexStepMode::Instance;
    for (uint32_t column = 0; column < 4; ++column)
        transform.attributes.push_back(
            {Backend::VertexFormat::Float32x4, column * sizeof(glm::vec4),
             static_cast<uint32_t>(RendererAttribute::InstanceTransform0) +
                 column});
    Backend::VertexBufferLayout normal;
    normal.arrayStride = sizeof(glm::vec3);
    normal.attributes = {
        {Backend::VertexFormat::Float32x3, 0, RendererAttribute::Normal}};
    Backend::VertexBufferLayout color;
    color.arrayStride = sizeof(glm::vec4);
    color.stepMode = Backend::VertexStepMode::Instance;
    color.attributes = {{Backend::VertexFormat::Float32x4, 0,
                         RendererAttribute::InstanceColor}};
    Backend::VertexBufferLayout empty;

    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "forward_vertex_color_pipeline";
    pipelineDesc.shader.name = "forward_vertex_color_rhi";
    pipelineDesc.shader.stages = {
        {_shaderLibrary->load(
             KE::getAssetPath("shaders/rhi/forward_vertex_color.vs")),
         Backend::ShaderType::Vertex, "main"},
        {_shaderLibrary->load(
             KE::getAssetPath("shaders/rhi/forward_vertex_color.fs")),
         Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.pipelineLayout = _forwardPipelineLayout;
    pipelineDesc.vertexBuffers = {position, transform, normal, empty, color};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    pipelineDesc.depthStencil =
        Backend::DepthStencilState{Backend::TextureFormat::Depth24Stencil8,
                                   true, Backend::CompareFunction::Less};
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.sampleCount = 4;
    createPipeline({RasterPipelineFamily::VertexColor, false, false, false},
                   pipelineDesc);
    pipelineDesc.label = "forward_vertex_color_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    createPipeline({RasterPipelineFamily::VertexColor, false, false, true},
                   pipelineDesc);

    Backend::BindGroupLayoutDesc skinLayoutDesc;
    skinLayoutDesc.label = "forward_skin_group_layout";
    skinLayoutDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                               Backend::ShaderStageVisibility::Vertex}};
    _skinGroupLayout = createBindGroupLayout(skinLayoutDesc);
    Backend::PipelineLayoutDesc skinPipelineLayoutDesc;
    skinPipelineLayoutDesc.label = "forward_skin_pipeline_layout";
    skinPipelineLayoutDesc.bindGroupLayouts = {
        _groupLayouts[0], shadowSamplingLayout, _skinGroupLayout,
        _groupLayouts[2]};
    _skinPipelineLayout = createPipelineLayout(skinPipelineLayoutDesc);
    Backend::VertexBufferLayout boneIndices;
    boneIndices.arrayStride = sizeof(glm::ivec4);
    boneIndices.attributes = {
        {Backend::VertexFormat::Sint32x4, 0, RendererAttribute::BoneIndices}};
    Backend::VertexBufferLayout boneWeights;
    boneWeights.arrayStride = sizeof(glm::vec4);
    boneWeights.attributes = {
        {Backend::VertexFormat::Float32x4, 0, RendererAttribute::BoneWeights}};
    pipelineDesc.label = "forward_skinned_vertex_color_pipeline";
    pipelineDesc.shader.name = "forward_skinned_vertex_color_rhi";
    pipelineDesc.shader.stages[0] = {
        _shaderLibrary->load(
            KE::getAssetPath("shaders/rhi/forward_skinned_vertex_color.vs")),
        Backend::ShaderType::Vertex, "main"};
    pipelineDesc.pipelineLayout = _skinPipelineLayout;
    pipelineDesc.vertexBuffers = {position,    transform,  normal,
                                  empty,       color,      empty,
                                  boneIndices, boneWeights};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    createPipeline({RasterPipelineFamily::VertexColor, true, false, false},
                   pipelineDesc);
    pipelineDesc.label = "forward_skinned_vertex_color_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    createPipeline({RasterPipelineFamily::VertexColor, true, false, true},
                   pipelineDesc);

    // Phong Material Pipeline
    Backend::BindGroupLayoutDesc materialLayoutDesc;
    materialLayoutDesc.label = "phong_material_group_layout";
    materialLayoutDesc.entries = {
        {0, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
        {1, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {3, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {4, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {5, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
    };
    _phongMaterialGroupLayout = createBindGroupLayout(materialLayoutDesc);
    Backend::PipelineLayoutDesc phongLayoutDesc;
    phongLayoutDesc.label = "forward_phong_pipeline_layout";
    phongLayoutDesc.bindGroupLayouts = {_groupLayouts[0], shadowSamplingLayout,
                                        _groupLayouts[1],
                                        _phongMaterialGroupLayout};
    _phongPipelineLayout = createPipelineLayout(phongLayoutDesc);

    Backend::SamplerDesc materialSamplerDesc;
    materialSamplerDesc.wrapU = Backend::TextureWrap::Repeat;
    materialSamplerDesc.wrapV = Backend::TextureWrap::Repeat;
    materialSamplerDesc.minFilter = Backend::TextureFilter::Linear;
    materialSamplerDesc.magFilter = Backend::TextureFilter::Linear;
    materialSamplerDesc.label = "forward_material_sampler";
    _materialSampler = createSampler(materialSamplerDesc);
    auto makeFallbackTexture = [&](const std::array<uint8_t, 4>& pixel,
                                   const char* label) {
        Backend::TextureResourceDesc desc;
        desc.extent = {1, 1, 1};
        desc.format = Backend::TextureFormat::RGBA8Unorm;
        desc.usage = Backend::TextureUsage::TextureBinding |
                     Backend::TextureUsage::CopyDst;
        desc.label = label;
        Backend::TextureInitialData initial{pixel.data(), pixel.size(), 4};
        return createTexture(desc, &initial);
    };
    _whiteTexture =
        makeFallbackTexture({255, 255, 255, 255}, "material_fallback_white");
    _normalTexture =
        makeFallbackTexture({128, 128, 255, 255}, "material_fallback_normal");

    Backend::VertexBufferLayout texCoord;
    texCoord.arrayStride = sizeof(glm::vec2);
    texCoord.attributes = {
        {Backend::VertexFormat::Float32x2, 0, RendererAttribute::TexCoord}};
    Backend::VertexBufferLayout tangent;
    tangent.arrayStride = sizeof(glm::vec4);
    tangent.attributes = {
        {Backend::VertexFormat::Float32x4, 0, RendererAttribute::Tangent}};

    // Textured Vertex-Color Pipeline
    // Covers commonTex.fs users such as SkinVisualBridge and deformable cloth
    // without routing them through the legacy texture-slot path.
    Backend::BindGroupLayoutDesc texturedLayoutDesc;
    texturedLayoutDesc.label = "textured_vertex_color_group_layout";
    texturedLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {1, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {3, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _texturedVertexColorGroupLayout = createBindGroupLayout(texturedLayoutDesc);
    for (size_t skin = 0; skin < 2; ++skin) {
        Backend::PipelineLayoutDesc texturedPipelineLayoutDesc;
        texturedPipelineLayoutDesc.label =
            skin ? "forward_skinned_textured_vertex_color_pipeline_layout"
                 : "forward_textured_vertex_color_pipeline_layout";
        texturedPipelineLayoutDesc.bindGroupLayouts = {
            _groupLayouts[0], shadowSamplingLayout,
            skin ? _skinGroupLayout : _groupLayouts[1],
            _texturedVertexColorGroupLayout};
        _texturedVertexColorPipelineLayouts[skin] =
            createPipelineLayout(texturedPipelineLayoutDesc);
    }
    Backend::BlendState texturedAlphaBlend;
    texturedAlphaBlend.color.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    texturedAlphaBlend.color.dstFactor =
        Backend::BlendFactorValue::OneMinusSrcAlpha;
    texturedAlphaBlend.alpha.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    texturedAlphaBlend.alpha.dstFactor =
        Backend::BlendFactorValue::OneMinusSrcAlpha;
    const std::string texturedFs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_textured_vertex_color.fs"));
    for (size_t skin = 0; skin < 2; ++skin) {
        const std::string texturedVs = _shaderLibrary->load(
            KE::getAssetPath(skin ? "shaders/rhi/forward_skinned_material.vs"
                                  : "shaders/rhi/forward_material.vs"));
        for (size_t transparent = 0; transparent < 2; ++transparent) {
            for (size_t doubleSided = 0; doubleSided < 2; ++doubleSided) {
                Backend::GraphicsPipelineDesc desc;
                desc.label = std::string("forward_") +
                             (skin ? "skinned_" : "") +
                             "textured_vertex_color_" +
                             (transparent ? "transparent_" : "opaque_") +
                             (doubleSided ? "double_sided" : "back_face");
                desc.shader.name = desc.label + "_rhi";
                desc.shader.stages = {
                    {texturedVs, Backend::ShaderType::Vertex, "main"},
                    {texturedFs, Backend::ShaderType::Fragment, "main"}};
                desc.pipelineLayout = _texturedVertexColorPipelineLayouts[skin];
                desc.vertexBuffers = {position, transform, normal,
                                      texCoord, color,     tangent};
                if (skin) {
                    desc.vertexBuffers.push_back(boneIndices);
                    desc.vertexBuffers.push_back(boneWeights);
                }
                desc.primitive.cullMode = doubleSided ? Backend::CullMode::None
                                                      : Backend::CullMode::Back;
                desc.depthStencil = Backend::DepthStencilState{
                    Backend::TextureFormat::Depth24Stencil8, !transparent,
                    Backend::CompareFunction::Less};
                desc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
                if (transparent)
                    desc.colorTargets[0].blend = texturedAlphaBlend;
                desc.sampleCount = 4;
                createPipeline({RasterPipelineFamily::TexturedVertexColor,
                                skin != 0, transparent != 0, doubleSided != 0},
                               desc);
            }
        }
    }

    // Checkerboard Ground Pipeline
    Backend::BindGroupLayoutDesc checkerLayoutDesc;
    checkerLayoutDesc.label = "checkerboard_group_layout";
    checkerLayoutDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                                  Backend::ShaderStageVisibility::Fragment}};
    _checkerboardGroupLayout = createBindGroupLayout(checkerLayoutDesc);
    Backend::PipelineLayoutDesc checkerPipelineLayoutDesc;
    checkerPipelineLayoutDesc.label = "forward_checkerboard_pipeline_layout";
    checkerPipelineLayoutDesc.bindGroupLayouts = {
        _groupLayouts[0], shadowSamplingLayout, _groupLayouts[1],
        _checkerboardGroupLayout};
    _checkerboardPipelineLayout =
        createPipelineLayout(checkerPipelineLayoutDesc);
    Backend::BufferDesc checkerParamsDesc;
    checkerParamsDesc.size = sizeof(glm::vec4) * 4;
    checkerParamsDesc.usage =
        Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
    checkerParamsDesc.label = "checkerboard_params";
    _checkerboardParamsBuffer = createBuffer(checkerParamsDesc);
    Backend::BindGroupDesc checkerGroupDesc;
    checkerGroupDesc.layout = _checkerboardGroupLayout;
    checkerGroupDesc.label = "checkerboard_bind_group";
    checkerGroupDesc.entries = {{0, _checkerboardParamsBuffer, 0,
                                 checkerParamsDesc.size, nullptr, nullptr}};
    _checkerboardBindGroup = createBindGroup(checkerGroupDesc);
    setBackgroundSettings(BackgroundSettings{});

    const std::string checkerVs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_material.vs"));
    const std::string checkerFs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_checkerboard.fs"));
    for (size_t transparent = 0; transparent < 2; ++transparent) {
        for (size_t doubleSided = 0; doubleSided < 2; ++doubleSided) {
            Backend::GraphicsPipelineDesc desc;
            desc.label = std::string("forward_checkerboard_") +
                         (transparent ? "transparent_" : "opaque_") +
                         (doubleSided ? "double_sided" : "back_face");
            desc.shader.name = desc.label + "_rhi";
            desc.shader.stages = {
                {checkerVs, Backend::ShaderType::Vertex, "main"},
                {checkerFs, Backend::ShaderType::Fragment, "main"}};
            desc.pipelineLayout = _checkerboardPipelineLayout;
            desc.vertexBuffers = {position, transform, normal,
                                  texCoord, color,     tangent};
            desc.primitive.cullMode =
                doubleSided ? Backend::CullMode::None : Backend::CullMode::Back;
            desc.depthStencil = Backend::DepthStencilState{
                Backend::TextureFormat::Depth24Stencil8, !transparent,
                Backend::CompareFunction::Less};
            desc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
            if (transparent)
                desc.colorTargets[0].blend = texturedAlphaBlend;
            desc.sampleCount = 4;
            createPipeline({RasterPipelineFamily::Checkerboard, false,
                            transparent != 0, doubleSided != 0},
                           desc);
        }
    }

    pipelineDesc.label = "forward_phong_pipeline";
    pipelineDesc.shader.name = "forward_phong_rhi";
    pipelineDesc.shader.stages = {
        {_shaderLibrary->load(
             KE::getAssetPath("shaders/rhi/forward_material.vs")),
         Backend::ShaderType::Vertex, "main"},
        {_shaderLibrary->load(KE::getAssetPath("shaders/rhi/forward_phong.fs")),
         Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.pipelineLayout = _phongPipelineLayout;
    pipelineDesc.vertexBuffers = {position, transform, normal,
                                  texCoord, color,     tangent};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    createPipeline({RasterPipelineFamily::Phong, false, false, false},
                   pipelineDesc);
    pipelineDesc.label = "forward_phong_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    createPipeline({RasterPipelineFamily::Phong, false, false, true},
                   pipelineDesc);

    // PBR Material Pipeline
    Backend::BindGroupLayoutDesc pbrMaterialLayoutDesc;
    pbrMaterialLayoutDesc.label = "pbr_material_group_layout";
    pbrMaterialLayoutDesc.entries.push_back(
        {0, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment});
    for (uint32_t binding = 1; binding <= 8; ++binding)
        pbrMaterialLayoutDesc.entries.push_back(
            {binding, Backend::BindingType::SampledTexture,
             Backend::ShaderStageVisibility::Fragment});
    pbrMaterialLayoutDesc.entries.push_back(
        {9, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment});
    _pbrMaterialGroupLayout = createBindGroupLayout(pbrMaterialLayoutDesc);
    Backend::PipelineLayoutDesc pbrLayoutDesc;
    pbrLayoutDesc.label = "forward_pbr_pipeline_layout";
    pbrLayoutDesc.bindGroupLayouts = {_groupLayouts[0], shadowSamplingLayout,
                                      _groupLayouts[1],
                                      _pbrMaterialGroupLayout};
    _pbrPipelineLayout = createPipelineLayout(pbrLayoutDesc);
    pipelineDesc.label = "forward_pbr_pipeline";
    pipelineDesc.shader.name = "forward_pbr_rhi";
    pipelineDesc.shader.stages[1] = {
        _shaderLibrary->load(KE::getAssetPath("shaders/rhi/forward_pbr.fs")),
        Backend::ShaderType::Fragment, "main"};
    pipelineDesc.pipelineLayout = _pbrPipelineLayout;
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    createPipeline({RasterPipelineFamily::Pbr, false, false, false},
                   pipelineDesc);
    pipelineDesc.label = "forward_pbr_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    createPipeline({RasterPipelineFamily::Pbr, false, false, true},
                   pipelineDesc);

    Backend::BlendState alphaBlend;
    alphaBlend.color.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    alphaBlend.color.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    alphaBlend.alpha.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    alphaBlend.alpha.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    auto createTransparentVariants =
        [&](RasterPipelineFamily family, bool skinned, const char* label,
            Backend::PipelineLayout* layout, const std::string& vertexSource,
            const std::string& fragmentSource,
            std::vector<Backend::VertexBufferLayout> buffers) {
            Backend::GraphicsPipelineDesc desc;
            desc.label = std::string(label) + "_pipeline";
            desc.shader.name = std::string(label) + "_rhi";
            desc.shader.stages = {
                {vertexSource, Backend::ShaderType::Vertex, "main"},
                {fragmentSource, Backend::ShaderType::Fragment, "main"}};
            desc.pipelineLayout = layout;
            desc.vertexBuffers = std::move(buffers);
            desc.depthStencil = Backend::DepthStencilState{
                Backend::TextureFormat::Depth24Stencil8, false,
                Backend::CompareFunction::Less};
            desc.colorTargets = {
                {Backend::TextureFormat::RGBA16Float, alphaBlend}};
            desc.sampleCount = 4;
            desc.primitive.cullMode = Backend::CullMode::Back;
            createPipeline({family, skinned, true, false}, desc);
            desc.label = std::string(label) + "_double_sided_pipeline";
            desc.primitive.cullMode = Backend::CullMode::None;
            createPipeline({family, skinned, true, true}, desc);
        };
    const std::string vertexColorVs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_vertex_color.vs"));
    const std::string skinnedVertexColorVs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_skinned_vertex_color.vs"));
    const std::string materialVs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_material.vs"));
    const std::string vertexColorFs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_vertex_color.fs"));
    const std::string phongFs =
        _shaderLibrary->load(KE::getAssetPath("shaders/rhi/forward_phong.fs"));
    const std::string pbrFs =
        _shaderLibrary->load(KE::getAssetPath("shaders/rhi/forward_pbr.fs"));
    createTransparentVariants(
        RasterPipelineFamily::VertexColor, false, "transparent_vertex_color",
        _forwardPipelineLayout, vertexColorVs, vertexColorFs,
        {position, transform, normal, empty, color});
    createTransparentVariants(RasterPipelineFamily::VertexColor, true,
                              "transparent_skinned_vertex_color",
                              _skinPipelineLayout, skinnedVertexColorVs,
                              vertexColorFs,
                              {position, transform, normal, empty, color, empty,
                               boneIndices, boneWeights});
    createTransparentVariants(
        RasterPipelineFamily::Phong, false, "transparent_phong",
        _phongPipelineLayout, materialVs, phongFs,
        {position, transform, normal, texCoord, color, tangent});
    createTransparentVariants(
        RasterPipelineFamily::Pbr, false, "transparent_pbr", _pbrPipelineLayout,
        materialVs, pbrFs,
        {position, transform, normal, texCoord, color, tangent});

    const std::string skinnedMaterialVs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_skinned_material.vs"));
    for (size_t model = 0; model < 2; ++model) {
        Backend::PipelineLayoutDesc skinnedLayoutDesc;
        skinnedLayoutDesc.label = model == 0
                                      ? "forward_skinned_phong_pipeline_layout"
                                      : "forward_skinned_pbr_pipeline_layout";
        skinnedLayoutDesc.bindGroupLayouts = {
            _groupLayouts[0], shadowSamplingLayout, _skinGroupLayout,
            model == 0 ? _phongMaterialGroupLayout : _pbrMaterialGroupLayout};
        _skinnedMaterialPipelineLayouts[model] =
            createPipelineLayout(skinnedLayoutDesc);
        for (size_t transparent = 0; transparent < 2; ++transparent) {
            Backend::GraphicsPipelineDesc desc;
            const std::string baseLabel =
                std::string("forward_skinned_") +
                (model == 0 ? "phong" : "pbr") +
                (transparent ? "_transparent" : "_opaque");
            desc.label = baseLabel + "_pipeline";
            desc.shader.name = baseLabel + "_rhi";
            desc.shader.stages = {
                {skinnedMaterialVs, Backend::ShaderType::Vertex, "main"},
                {model == 0 ? phongFs : pbrFs, Backend::ShaderType::Fragment,
                 "main"}};
            desc.pipelineLayout = _skinnedMaterialPipelineLayouts[model];
            desc.vertexBuffers = {position,    transform,  normal,
                                  texCoord,    color,      tangent,
                                  boneIndices, boneWeights};
            desc.depthStencil = Backend::DepthStencilState{
                Backend::TextureFormat::Depth24Stencil8, !transparent,
                Backend::CompareFunction::Less};
            desc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
            if (transparent)
                desc.colorTargets[0].blend = alphaBlend;
            desc.sampleCount = 4;
            desc.primitive.cullMode = Backend::CullMode::Back;
            createPipeline({model == 0 ? RasterPipelineFamily::Phong
                                       : RasterPipelineFamily::Pbr,
                            true, transparent != 0, false},
                           desc);
            desc.label = baseLabel + "_double_sided_pipeline";
            desc.primitive.cullMode = Backend::CullMode::None;
            createPipeline({model == 0 ? RasterPipelineFamily::Phong
                                       : RasterPipelineFamily::Pbr,
                            true, transparent != 0, true},
                           desc);
        }
    }

    const std::string debugCheckerFs = _shaderLibrary->load(
        KE::getAssetPath("shaders/rhi/forward_debug_checker.fs"));
    for (size_t skin = 0; skin < 2; ++skin) {
        for (size_t transparent = 0; transparent < 2; ++transparent) {
            Backend::GraphicsPipelineDesc desc;
            const std::string baseLabel =
                std::string("forward_") + (skin ? "skinned_" : "") +
                "debug_checker_" + (transparent ? "transparent" : "opaque");
            desc.label = baseLabel + "_pipeline";
            desc.shader.name = baseLabel + "_rhi";
            desc.shader.stages = {
                {skin ? skinnedMaterialVs : materialVs,
                 Backend::ShaderType::Vertex, "main"},
                {debugCheckerFs, Backend::ShaderType::Fragment, "main"}};
            desc.pipelineLayout =
                skin ? _skinPipelineLayout : _forwardPipelineLayout;
            desc.vertexBuffers = {position, transform, normal, texCoord, color};
            if (skin) {
                // Preserve the shared mesh slot numbering without requiring
                // the unused tangent stream.
                desc.vertexBuffers.push_back({});
                desc.vertexBuffers.push_back(boneIndices);
                desc.vertexBuffers.push_back(boneWeights);
            }
            desc.depthStencil = Backend::DepthStencilState{
                Backend::TextureFormat::Depth24Stencil8, !transparent,
                Backend::CompareFunction::Less};
            desc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
            if (transparent)
                desc.colorTargets[0].blend = alphaBlend;
            desc.sampleCount = 4;
            desc.primitive.cullMode = Backend::CullMode::Back;
            createPipeline({RasterPipelineFamily::DebugChecker, skin != 0,
                            transparent != 0, false},
                           desc);
            desc.label = baseLabel + "_double_sided_pipeline";
            desc.primitive.cullMode = Backend::CullMode::None;
            createPipeline({RasterPipelineFamily::DebugChecker, skin != 0,
                            transparent != 0, true},
                           desc);
        }
    }
    configureMaterialBindings(_texturedVertexColorGroupLayout,
                              _phongMaterialGroupLayout,
                              _pbrMaterialGroupLayout, _materialSampler,
                              _whiteTexture, _normalTexture);
}

void ForwardPass::setBackgroundSettings(const BackgroundSettings& settings) {
    if (!_checkerboardParamsBuffer)
        return;
    const std::array<glm::vec4, 4> params{
        settings.checkerColor1, settings.checkerColor2, settings.gridColor,
        glm::vec4(settings.showGrid ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f)};
    _checkerboardParamsBuffer->setData(params.data(), sizeof(params));
}

} // namespace KE

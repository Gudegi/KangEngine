#include "engine/graphics/renderer/picking_pass.hpp"

#include "engine/graphics/renderer/mesh_instancer.hpp"
#include "engine/graphics/renderer/shader_library.hpp"
#include "utils/asset_path.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace KE {

void PickingPass::initializeResources() {
    using namespace Backend;
    requireInitialized("creating resources");
    BindGroupLayoutDesc layout;
    layout.entries = {
        {0, BindingType::UniformBuffer, ShaderStageVisibility::Vertex}};
    _layouts[0] = _device->createBindGroupLayout(layout);
    layout.entries[0].visibility =
        ShaderStageVisibility::Vertex | ShaderStageVisibility::Fragment;
    _layouts[1] = _device->createBindGroupLayout(layout);
    layout.entries[0].visibility = ShaderStageVisibility::Vertex;
    _layouts[2] = _device->createBindGroupLayout(layout);
    layout.entries = {
        {0, BindingType::SampledTexture, ShaderStageVisibility::Fragment},
        {1, BindingType::Sampler, ShaderStageVisibility::Fragment}};
    _layouts[3] = _device->createBindGroupLayout(layout);
    _emptyLayout = _device->createBindGroupLayout({});
    for (int skin = 0; skin < 2; ++skin) {
        for (int textured = 0; textured < 2; ++textured) {
            PipelineLayoutDesc pipelineLayout;
            pipelineLayout.bindGroupLayouts = {
                _layouts[0].get(), _layouts[1].get(),
                skin ? _layouts[2].get() : _emptyLayout.get(),
                textured ? _layouts[3].get() : _emptyLayout.get()};
            _pipelineLayouts[skin * 2 + textured] =
                _device->createPipelineLayout(pipelineLayout);
        }
    }

    BufferDesc buffer;
    buffer.size = sizeof(glm::mat4);
    buffer.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
    buffer.label = "picking_camera";
    _camera = _device->createBuffer(buffer);
    BindGroupDesc group;
    group.layout = _layouts[0].get();
    group.entries = {{0, _camera.get(), 0, buffer.size, nullptr, nullptr}};
    _frame = _device->createBindGroup(group);
    SamplerDesc sampler;
    sampler.minFilter = TextureFilter::Linear;
    sampler.magFilter = TextureFilter::Linear;
    sampler.wrapU = sampler.wrapV = TextureWrap::Repeat;
    _sampler = _device->createSampler(sampler);

    TextureResourceDesc texture;
    texture.extent = {1, 1, 1};
    texture.usage = TextureUsage::RenderAttachment | TextureUsage::CopySrc;
    texture.format = TextureFormat::RGBA8Unorm;
    _color = _device->createTexture(texture);
    texture.format = TextureFormat::Depth32Float;
    _depth = _device->createTexture(texture);
    _colorView = _device->createTextureView(_color.get(), {});
    TextureViewDesc depthView;
    depthView.aspect = TextureAspect::DepthOnly;
    _depthView = _device->createTextureView(_depth.get(), depthView);
    RenderPassDesc target;
    target.label = "picking";
    target.colorAttachments = {{_colorView.get(),
                                nullptr,
                                LoadOp::Clear,
                                StoreOp::Store,
                                {0, 0, 0, 0}}};
    DepthStencilAttachmentDesc depth;
    depth.view = _depthView.get();
    depth.depthLoadOp = LoadOp::Clear;
    target.depthStencilAttachment = depth;
    _target = _device->createRenderTarget(target);
}

Backend::GraphicsPipeline* PickingPass::pipeline(ShaderLibrary& shaders,
                                                 bool skinned, bool textured,
                                                 bool doubleSided) {
    using namespace Backend;
    auto& result = _pipelines[(skinned ? 4 : 0) | (textured ? 2 : 0) |
                              (doubleSided ? 1 : 0)];
    if (result)
        return result.get();
    GraphicsPipelineDesc desc;
    desc.label = "picking";
    desc.shader.name = "picking";
    std::string vs = shaders.load(getAssetPath("shaders/rhi/picking.vs"));
    std::string fs = shaders.load(getAssetPath("shaders/rhi/picking.fs"));
    if (skinned)
        vs.insert(vs.find('\n') + 1, "#define KE_SKINNED\n");
    if (textured)
        fs.insert(fs.find('\n') + 1, "#define KE_TEXTURED\n");
    desc.shader.stages = {{vs, ShaderType::Vertex, "main"},
                          {fs, ShaderType::Fragment, "main"}};
    desc.pipelineLayout =
        _pipelineLayouts[(skinned ? 2 : 0) | (textured ? 1 : 0)].get();
    desc.vertexBuffers.resize(8);
    desc.vertexBuffers[0] = {
        sizeof(glm::vec3),
        VertexStepMode::Vertex,
        {{VertexFormat::Float32x3, 0, RendererAttribute::Position}}};
    auto& transforms = desc.vertexBuffers[1];
    transforms.arrayStride = sizeof(glm::mat4);
    transforms.stepMode = VertexStepMode::Instance;
    for (uint32_t i = 0; i < 4; ++i)
        transforms.attributes.push_back(
            {VertexFormat::Float32x4, i * sizeof(glm::vec4),
             RendererAttribute::InstanceTransform0 + i});
    if (textured)
        desc.vertexBuffers[3] = {
            sizeof(glm::vec2),
            VertexStepMode::Vertex,
            {{VertexFormat::Float32x2, 0, RendererAttribute::TexCoord}}};
    desc.vertexBuffers[4] = {
        sizeof(float),
        VertexStepMode::Instance,
        {{VertexFormat::Float32, 0, RendererAttribute::InstanceColor}}};
    if (skinned) {
        desc.vertexBuffers[6] = {
            sizeof(glm::ivec4),
            VertexStepMode::Vertex,
            {{VertexFormat::Sint32x4, 0, RendererAttribute::BoneIndices}}};
        desc.vertexBuffers[7] = {
            sizeof(glm::vec4),
            VertexStepMode::Vertex,
            {{VertexFormat::Float32x4, 0, RendererAttribute::BoneWeights}}};
    }
    desc.primitive.cullMode = doubleSided ? CullMode::None : CullMode::Back;
    desc.depthStencil = DepthStencilState{};
    desc.colorTargets = {{TextureFormat::RGBA8Unorm}};
    result = _device->createGraphicsPipeline(desc);
    return result.get();
}

RayPickResult PickingPass::pick(const std::vector<MeshInstancer*>& instancers,
                                ShaderLibrary& shaders, const glm::mat4& view,
                                const glm::mat4& projection, glm::vec2 ndc,
                                int width, int height) {
    using namespace Backend;
    if (width <= 0 || height <= 0 || !std::isfinite(ndc.x) ||
        !std::isfinite(ndc.y) || ndc.x < -1 || ndc.x >= 1 || ndc.y <= -1 ||
        ndc.y > 1)
        return {};
    if (!_target)
        initializeResources();

    // Crop one viewport pixel into a 1x1 target; retain the original depth
    // mapping.
    const int x = std::min(width - 1, int((ndc.x + 1) * 0.5f * width));
    const int y = std::min(height - 1, int((ndc.y + 1) * 0.5f * height));
    ndc = {2.0f * (x + 0.5f) / width - 1, 2.0f * (y + 0.5f) / height - 1};
    glm::mat4 crop(1.0f);
    crop[0][0] = float(width);
    crop[1][1] = float(height);
    crop[3][0] = -ndc.x * width;
    crop[3][1] = -ndc.y * height;
    const glm::mat4 vp = crop * projection * view;
    _camera->setData(&vp, sizeof(vp));

    struct Draw {
        MeshInstancer* instancer;
        RenderableHandle handle;
        uint32_t firstId;
        MeshInstancer::PickInstances instances;
        GraphicsPipeline* pipeline;
        std::unique_ptr<Buffer> params;
        std::unique_ptr<BindGroup> batch;
        std::unique_ptr<BindGroup> skin;
        std::unique_ptr<TextureView> texture;
        std::unique_ptr<BindGroup> alpha;
    };
    std::vector<Draw> draws;
    uint64_t nextId = 1;
    for (size_t handle = 0; handle < instancers.size(); ++handle) {
        auto* inst = instancers[handle];
        if (!inst)
            continue;
        auto instances = inst->preparePickInstances();
        if (!instances.count)
            continue;
        if (nextId + instances.count >
            uint64_t(std::numeric_limits<uint32_t>::max()))
            throw std::overflow_error("Picking instance IDs exceed 32 bits");
        Draw draw{};
        draw.instancer = inst;
        draw.handle = static_cast<RenderableHandle>(handle);
        draw.firstId = static_cast<uint32_t>(nextId);
        nextId += instances.count;
        draw.instances = std::move(instances);
        Material* material = inst->material();
        const bool useAlpha =
            inst->alphaMode() == AlphaMode::Mask || inst->hasTransparent();
        Texture* texture = material->alphaTexture();
        if (material->shadingModel() == MaterialShadingModel::Color &&
            (material->isUnlit() ||
             material->colorStyle() == ColorStyle::Textured))
            texture = inst->textureAtSlot(RendererTextureSlot::BaseColor);
        if (!useAlpha)
            texture = nullptr;
        draw.pipeline = pipeline(shaders, inst->hasSkinning(),
                                 texture != nullptr, inst->isDoubleSided());
        struct Params {
            glm::uvec4 ids;
            glm::vec4 alpha;
        } params;
        params.ids = {draw.firstId, 0, 0, 0};
        params.alpha = {
            inst->alphaMode() == AlphaMode::Mask ? inst->alphaCutoff() : -1.0f,
            inst->alphaMaskUsesRedChannel() ? 1.0f : 0.0f, 1.0f, 0.0f};
        if (useAlpha) {
            if (auto* pbr = dynamic_cast<PBRMaterial*>(material))
                params.alpha.z = pbr->baseColor.a;
        }
        BufferDesc buffer;
        buffer.size = sizeof(params);
        buffer.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
        draw.params = _device->createBuffer(buffer);
        draw.params->setData(&params, sizeof(params));
        BindGroupDesc group;
        group.layout = _layouts[1].get();
        group.entries = {
            {0, draw.params.get(), 0, sizeof(params), nullptr, nullptr}};
        draw.batch = _device->createBindGroup(group);
        if (inst->hasSkinning()) {
            group.layout = _layouts[2].get();
            group.entries = {{0, inst->boneMatricesBuffer(), 0,
                              sizeof(glm::mat4) * Scene::MaxSkinningBones,
                              nullptr, nullptr}};
            draw.skin = _device->createBindGroup(group);
        }
        if (texture) {
            draw.texture = _device->createTextureView(texture, {});
            group.layout = _layouts[3].get();
            group.entries = {{0, nullptr, 0, 0, draw.texture.get(), nullptr},
                             {1, nullptr, 0, 0, nullptr, _sampler.get()}};
            draw.alpha = _device->createBindGroup(group);
        }
        draws.push_back(std::move(draw));
    }

    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(_target.get());
    pass->setViewport(0, 0, 1, 1);
    pass->setScissor(0, 0, 1, 1);
    pass->setPolygonMode(PolygonMode::Fill);
    for (const auto& draw : draws) {
        pass->setPipeline(draw.pipeline);
        pass->setBindGroup(0, _frame.get());
        pass->setBindGroup(1, draw.batch.get());
        if (draw.skin)
            pass->setBindGroup(2, draw.skin.get());
        if (draw.alpha)
            pass->setBindGroup(3, draw.alpha.get());
        draw.instancer->recordPickDraw(*pass, draw.instances,
                                       draw.alpha != nullptr);
    }
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
    const auto color = _device->readTexture(_colorView.get());
    uint32_t id = 0;
    for (uint32_t i = 0; i < 4; ++i)
        id |= static_cast<uint32_t>(std::lround(color.values[i] * 255.0f))
              << (8 * i);
    if (!id)
        return {};

    const float depth = _device->readTexture(_depthView.get()).values[0];
    const glm::mat4 inverse = glm::inverse(projection * view);
    glm::vec4 world = inverse * glm::vec4(ndc, depth * 2 - 1, 1);
    world /= world.w;
    glm::vec4 near = inverse * glm::vec4(ndc, -1, 1);
    near /= near.w;
    for (const auto& draw : draws) {
        if (id < draw.firstId || id - draw.firstId >= draw.instances.count)
            continue;
        auto result = draw.instancer->pickInstance(int(id - draw.firstId));
        result.handle = draw.handle;
        result.position = glm::vec3(world);
        result.distance = glm::length(glm::vec3(world - near));
        return result;
    }
    return {};
}

} // namespace KE

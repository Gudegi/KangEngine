#include "engine/graphics/renderer/skybox_pass.hpp"

#include "utils/asset_path.hpp"

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace KE {

void SkyboxPass::initializeResources(Backend::BindGroupLayout* frameLayout) {
    requireInitialized("initializing resources");
    if (!frameLayout)
        throw std::invalid_argument("SkyboxPass requires a frame layout");

    static constexpr glm::vec3 vertices[] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    static constexpr uint32_t indices[] = {0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1,
                                           5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4,
                                           3, 2, 6, 6, 7, 3, 4, 5, 1, 1, 0, 4};
    Backend::BufferDesc bufferDesc;
    bufferDesc.size = sizeof(vertices);
    bufferDesc.usage = Backend::BufferUsage::Vertex;
    bufferDesc.label = "skybox_vertices";
    _vertexBuffer = _device->createBuffer(bufferDesc, vertices);
    bufferDesc.size = sizeof(indices);
    bufferDesc.usage = Backend::BufferUsage::Index;
    bufferDesc.label = "skybox_indices";
    _indexBuffer = _device->createBuffer(bufferDesc, indices);
    bufferDesc.size = sizeof(glm::vec4);
    bufferDesc.usage =
        Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
    bufferDesc.label = "skybox_params";
    _paramsBuffer = _device->createBuffer(bufferDesc);

    Backend::BindGroupLayoutDesc passLayoutDesc;
    passLayoutDesc.label = "skybox_pass_group_layout";
    passLayoutDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                               Backend::ShaderStageVisibility::Vertex}};
    _passGroupLayout = _device->createBindGroupLayout(passLayoutDesc);
    Backend::BindGroupDesc paramsDesc;
    paramsDesc.layout = _passGroupLayout.get();
    paramsDesc.label = "skybox_params_bind_group";
    paramsDesc.entries = {
        {0, _paramsBuffer.get(), 0, sizeof(glm::vec4), nullptr, nullptr}};
    _paramsBindGroup = _device->createBindGroup(paramsDesc);

    Backend::BindGroupLayoutDesc textureLayoutDesc;
    textureLayoutDesc.label = "skybox_texture_group_layout";
    textureLayoutDesc.entries = {{0, Backend::BindingType::SampledTexture,
                                  Backend::ShaderStageVisibility::Fragment,
                                  Backend::TextureFormat::Undefined,
                                  Backend::TextureSampleType::Float,
                                  Backend::TextureViewDimension::Cube},
                                 {1, Backend::BindingType::Sampler,
                                  Backend::ShaderStageVisibility::Fragment}};
    _textureGroupLayout = _device->createBindGroupLayout(textureLayoutDesc);

    Backend::BindGroupLayoutDesc emptyDesc;
    emptyDesc.label = "skybox_reserved_group_layout";
    auto reservedLayout = _device->createBindGroupLayout(emptyDesc);
    Backend::PipelineLayoutDesc pipelineLayoutDesc;
    pipelineLayoutDesc.label = "skybox_pipeline_layout";
    pipelineLayoutDesc.bindGroupLayouts = {frameLayout, _passGroupLayout.get(),
                                           reservedLayout.get(),
                                           _textureGroupLayout.get()};
    _pipelineLayout = _device->createPipelineLayout(pipelineLayoutDesc);

    Backend::VertexBufferLayout vertexLayout;
    vertexLayout.arrayStride = sizeof(glm::vec3);
    vertexLayout.attributes = {{Backend::VertexFormat::Float32x3, 0, 0}};
    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "skybox_pipeline";
    pipelineDesc.shader.name = "skybox_rhi";
    pipelineDesc.shader.stages = {
        {Backend::loadShaderSource(KE::getAssetPath("shaders/rhi/skybox.vs")),
         Backend::ShaderType::Vertex, "main"},
        {Backend::loadShaderSource(KE::getAssetPath("shaders/rhi/skybox.fs")),
         Backend::ShaderType::Fragment, "main"}};
    pipelineDesc.pipelineLayout = _pipelineLayout.get();
    pipelineDesc.vertexBuffers = {vertexLayout};
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.depthStencil =
        Backend::DepthStencilState{Backend::TextureFormat::Depth24Stencil8,
                                   false, Backend::CompareFunction::LessEqual};
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.sampleCount = 4;
    _pipeline = _device->createGraphicsPipeline(pipelineDesc);

    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "skybox_sampler";
    _sampler = _device->createSampler(samplerDesc);
}

void SkyboxPass::setTexture(const std::string& path, UpAxis upAxis) {
    _texture = _device->createCubemapTexture(path);
    rebuildBinding(upAxis);
}

void SkyboxPass::setTexture(const std::vector<std::string>& paths,
                            UpAxis upAxis) {
    _texture = _device->createCubemapTexture(paths);
    rebuildBinding(upAxis);
}

void SkyboxPass::rebuildBinding(UpAxis upAxis) {
    const glm::vec4 params{upAxis == UpAxis::Z ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f};
    _paramsBuffer->setData(&params, sizeof(params));
    _textureBindGroup.reset();
    _textureView.reset();
    if (!_texture)
        return;
    Backend::TextureViewDesc viewDesc;
    viewDesc.dimension = Backend::TextureViewDimension::Cube;
    viewDesc.arrayLayerCount = 6;
    viewDesc.label = "skybox_cube_view";
    _textureView = _device->createTextureView(_texture.get(), viewDesc);
    Backend::BindGroupDesc groupDesc;
    groupDesc.layout = _textureGroupLayout.get();
    groupDesc.label = "skybox_texture_bind_group";
    groupDesc.entries = {{0, nullptr, 0, 0, _textureView.get(), nullptr},
                         {1, nullptr, 0, 0, nullptr, _sampler.get()}};
    _textureBindGroup = _device->createBindGroup(groupDesc);
}

void SkyboxPass::record(Backend::RenderPassEncoder& pass,
                        Backend::BindGroup* frameBindGroup) const {
    if (!ready())
        return;
    pass.setPipeline(_pipeline.get());
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, _paramsBindGroup.get());
    pass.setBindGroup(3, _textureBindGroup.get());
    pass.setVertexBuffer(0, _vertexBuffer.get());
    pass.setIndexBuffer(_indexBuffer.get(), Backend::IndexFormat::Uint32);
    pass.drawIndexed(36);
}

} // namespace KE

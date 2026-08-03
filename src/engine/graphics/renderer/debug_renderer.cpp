#include "engine/graphics/renderer/debug_renderer.hpp"
#include "engine/graphics/material/colors.hpp"
#include "utils/asset_path.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace KE {
namespace {

glm::vec4 pickColor(const std::vector<glm::vec4>& colors, size_t i) {
    if (colors.empty())
        return glm::vec4(1.0f);
    if (colors.size() == 1)
        return colors.front();
    return colors[std::min(i, colors.size() - 1)];
}

glm::vec4 presetColor(ColorType type) {
    const Color& c = ColorLibrary::get(type);
    return glm::vec4(c.r, c.g, c.b, c.a);
}

size_t growVertexCapacity(size_t requested) {
    if (requested == 0)
        return 0;
    return std::max<size_t>(requested * 2, 8);
}

void validateLineInputs(const char* functionName,
                        const std::vector<glm::vec3>& starts,
                        const std::vector<glm::vec3>& ends,
                        const std::vector<glm::vec4>& colors) {
    if (starts.size() != ends.size()) {
        throw std::invalid_argument(std::string(functionName) +
                                    " requires starts and ends to have the "
                                    "same length.");
    }
    if (!colors.empty() && colors.size() != 1 &&
        colors.size() != starts.size()) {
        throw std::invalid_argument(std::string(functionName) +
                                    " colors must be empty, length 1, or "
                                    "match the number of lines.");
    }
}

void validatePointInputs(const char* functionName,
                         const std::vector<glm::vec3>& points,
                         const std::vector<glm::vec4>& colors) {
    if (!colors.empty() && colors.size() != 1 &&
        colors.size() != points.size()) {
        throw std::invalid_argument(std::string(functionName) +
                                    " colors must be empty, length 1, or "
                                    "match the number of points.");
    }
}

} // namespace

void DebugRenderer::init(Backend::GraphicsDevice* device,
                         Backend::BindGroupLayout* frameGroupLayout,
                         Backend::BindGroup* frameBindGroup) {
    _device = device;
    if (!_device)
        return;

    _frameBindGroup = frameBindGroup;
    if (!frameGroupLayout || !_frameBindGroup)
        return;
    for (size_t i = 0; i < _reservedGroupLayouts.size(); ++i) {
        Backend::BindGroupLayoutDesc desc;
        desc.label = "debug_reserved_group_" + std::to_string(i + 1);
        _reservedGroupLayouts[i] = _device->createBindGroupLayout(desc);
    }
    Backend::PipelineLayoutDesc layoutDesc;
    layoutDesc.label = "debug_pipeline_layout";
    layoutDesc.bindGroupLayouts = {
        frameGroupLayout, _reservedGroupLayouts[0].get(),
        _reservedGroupLayouts[1].get(), _reservedGroupLayouts[2].get()};
    _pipelineLayout = _device->createPipelineLayout(layoutDesc);

    Backend::BlendState blend;
    blend.color.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    blend.color.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    blend.alpha.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    blend.alpha.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    Backend::VertexBufferLayout lineLayout;
    lineLayout.arrayStride = sizeof(LineVertex);
    lineLayout.attributes = {
        {Backend::VertexFormat::Float32x3, offsetof(LineVertex, position), 0},
        {Backend::VertexFormat::Float32x4, offsetof(LineVertex, color), 1}};
    Backend::GraphicsPipelineDesc desc;
    desc.label = "debug_line_pipeline";
    desc.shader.name = "debug_line_rhi";
    desc.shader.stages = {
        {Backend::loadShaderSource(KE::getAssetPath("shaders/rhi/debug.vs")),
         Backend::ShaderType::Vertex, "main"},
        {Backend::loadShaderSource(
             KE::getAssetPath("shaders/rhi/debug_line.fs")),
         Backend::ShaderType::Fragment, "main"}};
    desc.pipelineLayout = _pipelineLayout.get();
    desc.vertexBuffers = {lineLayout};
    desc.primitive.topology = Backend::PrimitiveTopology::LineList;
    desc.primitive.cullMode = Backend::CullMode::None;
    desc.depthStencil = Backend::DepthStencilState{
        Backend::TextureFormat::Depth24Stencil8, true,
        Backend::CompareFunction::Less};
    desc.colorTargets = {{Backend::TextureFormat::RGBA16Float, blend}};
    desc.sampleCount = 4;
    _linePipeline = _device->createGraphicsPipeline(desc);

    Backend::VertexBufferLayout pointLayout;
    pointLayout.arrayStride = sizeof(PointVertex);
    pointLayout.attributes = {
        {Backend::VertexFormat::Float32x3, offsetof(PointVertex, position), 0},
        {Backend::VertexFormat::Float32x4, offsetof(PointVertex, color), 1},
        {Backend::VertexFormat::Float32, offsetof(PointVertex, size), 2}};
    desc.label = "debug_point_pipeline";
    desc.shader.name = "debug_point_rhi";
    desc.shader.stages[1] =
        {Backend::loadShaderSource(
             KE::getAssetPath("shaders/rhi/debug_point.fs")),
         Backend::ShaderType::Fragment, "main"};
    desc.vertexBuffers = {pointLayout};
    desc.primitive.topology = Backend::PrimitiveTopology::PointList;
    _pointPipeline = _device->createGraphicsPipeline(desc);
}

void DebugRenderer::ensureLineBatchGpu(LineBatch& batch) {
    if (!_device)
        return;

    const size_t bytes = batch.vertices.size() * sizeof(LineVertex);
    if (batch.vertices.size() > batch.allocatedVertices) {
        batch.allocatedVertices = growVertexCapacity(batch.vertices.size());
        Backend::BufferDesc desc;
        desc.size = batch.allocatedVertices * sizeof(LineVertex);
        desc.usage = Backend::BufferUsage::Vertex |
                     Backend::BufferUsage::CopyDst;
        desc.label = "debug_line_vertices";
        batch.vertexBuffer = _device->createBuffer(desc);
    }

    if (bytes > 0)
        batch.vertexBuffer->setData(batch.vertices.data(), bytes);
}

void DebugRenderer::ensurePointBatchGpu(PointBatch& batch) {
    if (!_device)
        return;

    const size_t bytes = batch.vertices.size() * sizeof(PointVertex);
    if (batch.vertices.size() > batch.allocatedVertices) {
        batch.allocatedVertices = growVertexCapacity(batch.vertices.size());
        Backend::BufferDesc desc;
        desc.size = batch.allocatedVertices * sizeof(PointVertex);
        desc.usage = Backend::BufferUsage::Vertex |
                     Backend::BufferUsage::CopyDst;
        desc.label = "debug_point_vertices";
        batch.vertexBuffer = _device->createBuffer(desc);
    }

    if (bytes > 0)
        batch.vertexBuffer->setData(batch.vertices.data(), bytes);
}

void DebugRenderer::logLines(const std::string& path,
                             const std::vector<glm::vec3>& starts,
                             const std::vector<glm::vec3>& ends,
                             const std::vector<glm::vec4>& colors, float width,
                             bool hidden) {
    if (path.empty())
        return;
    validateLineInputs("DebugRenderer::logLines", starts, ends, colors);

    auto& batch = _lineBatches[path];
    batch.vertices.clear();
    batch.vertices.reserve(starts.size() * 2);
    batch.width = std::max(1.0f, width);
    batch.hidden = hidden;

    for (size_t i = 0; i < starts.size(); ++i) {
        const glm::vec4 color = pickColor(colors, i);
        batch.vertices.push_back({starts[i], color});
        batch.vertices.push_back({ends[i], color});
    }

    ensureLineBatchGpu(batch);
}

void DebugRenderer::logAxes(const std::string& path, const glm::mat4& transform,
                            float length, float width, bool hidden) {
    logAxes(path, glm::vec3(transform[3]), glm::vec3(transform[0]),
            glm::vec3(transform[1]), glm::vec3(transform[2]), length, width,
            hidden);
}

void DebugRenderer::logAxes(const std::string& path, const glm::vec3& origin,
                            const glm::vec3& xAxis, const glm::vec3& yAxis,
                            const glm::vec3& zAxis, float length, float width,
                            bool hidden) {
    const float safeLength = std::max(0.0f, length);
    const std::vector<glm::vec3> starts{origin, origin, origin};
    const std::vector<glm::vec3> ends{
        origin + glm::normalize(xAxis) * safeLength,
        origin + glm::normalize(yAxis) * safeLength,
        origin + glm::normalize(zAxis) * safeLength,
    };
    const std::vector<glm::vec4> colors{
        presetColor(ColorType::RED),
        presetColor(ColorType::GREEN),
        presetColor(ColorType::BLUE),
    };
    logLines(path, starts, ends, colors, width, hidden);
}

void DebugRenderer::clearLines(const std::string& path) {
    auto it = _lineBatches.find(path);
    if (it == _lineBatches.end())
        return;
    it->second.vertices.clear();
}

void DebugRenderer::logPoints(const std::string& path,
                              const std::vector<glm::vec3>& points,
                              const std::vector<glm::vec4>& colors, float size,
                              bool hidden) {
    if (path.empty())
        return;
    validatePointInputs("DebugRenderer::logPoints", points, colors);

    auto& batch = _pointBatches[path];
    batch.vertices.clear();
    batch.vertices.reserve(points.size());
    batch.hidden = hidden;

    const float safeSize = std::max(1.0f, size);
    for (size_t i = 0; i < points.size(); ++i)
        batch.vertices.push_back({points[i], pickColor(colors, i), safeSize});

    ensurePointBatchGpu(batch);
}

void DebugRenderer::clearPoints(const std::string& path) {
    auto it = _pointBatches.find(path);
    if (it == _pointBatches.end())
        return;
    it->second.vertices.clear();
}

void DebugRenderer::render(Backend::RenderTarget* target, int viewportWidth,
                           int viewportHeight) {
    if (!target || !_device || !_linePipeline || !_pointPipeline ||
        !_frameBindGroup || viewportWidth <= 0 || viewportHeight <= 0)
        return;
    bool hasRenderable = false;
    for (const auto& [path, batch] : _lineBatches)
        hasRenderable = hasRenderable ||
            (!batch.hidden && !batch.vertices.empty() && batch.vertexBuffer);
    for (const auto& [path, batch] : _pointBatches)
        hasRenderable = hasRenderable ||
            (!batch.hidden && !batch.vertices.empty() && batch.vertexBuffer);
    if (!hasRenderable)
        return;

    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(target);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(viewportWidth),
                      static_cast<float>(viewportHeight));
    for (auto& [path, batch] : _lineBatches) {
        if (batch.hidden || batch.vertices.empty() || !batch.vertexBuffer)
            continue;
        pass->setPipeline(_linePipeline.get());
        pass->setBindGroup(0, _frameBindGroup);
        pass->setLineWidth(batch.width);
        pass->setVertexBuffer(0, batch.vertexBuffer.get());
        pass->draw(static_cast<uint32_t>(batch.vertices.size()));
    }
    for (auto& [path, batch] : _pointBatches) {
        if (batch.hidden || batch.vertices.empty() || !batch.vertexBuffer)
            continue;
        pass->setPipeline(_pointPipeline.get());
        pass->setBindGroup(0, _frameBindGroup);
        pass->setVertexBuffer(0, batch.vertexBuffer.get());
        pass->draw(static_cast<uint32_t>(batch.vertices.size()));
    }
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
}

} // namespace KE

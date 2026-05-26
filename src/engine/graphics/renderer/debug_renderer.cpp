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

void DebugRenderer::init(Backend::GraphicsDevice* device) {
    _device = device;
    if (!_device)
        return;

    _ownedShader =
        _device->createShaderFromFile(KE::getAssetPath("shaders/debug.vs"),
                                      KE::getAssetPath("shaders/debug.fs"));
    _shader = _ownedShader.get();
    _shader->setUniformBlockBinding("cameraUBO", 0);
}

void DebugRenderer::ensureLineBatchGpu(LineBatch& batch) {
    if (!_device)
        return;

    const size_t bytes = batch.vertices.size() * sizeof(LineVertex);
    if (batch.vertices.size() > batch.allocatedVertices) {
        batch.allocatedVertices = growVertexCapacity(batch.vertices.size());
        batch.vao = _device->createVertexArray();
        batch.vertexBuffer =
            _device->createBuffer(Backend::BufferType::DynamicVertex,
                                  batch.allocatedVertices * sizeof(LineVertex));

        batch.vao->bind();
        batch.vao->setVertexBuffer(batch.vertexBuffer.get());
        batch.vao->setVertexAttribute(
            {0, 3, Backend::VertexAttributeType::Float, false,
             sizeof(LineVertex), offsetof(LineVertex, position), 0});
        batch.vao->setVertexAttribute(
            {1, 4, Backend::VertexAttributeType::Float, false,
             sizeof(LineVertex), offsetof(LineVertex, color), 0});
        batch.vao->unbind();
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
        batch.vao = _device->createVertexArray();
        batch.vertexBuffer = _device->createBuffer(
            Backend::BufferType::DynamicVertex,
            batch.allocatedVertices * sizeof(PointVertex));

        batch.vao->bind();
        batch.vao->setVertexBuffer(batch.vertexBuffer.get());
        batch.vao->setVertexAttribute(
            {0, 3, Backend::VertexAttributeType::Float, false,
             sizeof(PointVertex), offsetof(PointVertex, position), 0});
        batch.vao->setVertexAttribute(
            {1, 4, Backend::VertexAttributeType::Float, false,
             sizeof(PointVertex), offsetof(PointVertex, color), 0});
        batch.vao->setVertexAttribute(
            {2, 1, Backend::VertexAttributeType::Float, false,
             sizeof(PointVertex), offsetof(PointVertex, size), 0});
        batch.vao->unbind();
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

void DebugRenderer::render() {
    if (!_device)
        return;

    bool hasRenderable = false;
    for (auto& [path, batch] : _lineBatches) {
        if (!batch.hidden && !batch.vertices.empty() && batch.vao) {
            hasRenderable = true;
            break;
        }
    }
    if (!hasRenderable) {
        for (auto& [path, batch] : _pointBatches) {
            if (!batch.hidden && !batch.vertices.empty() && batch.vao) {
                hasRenderable = true;
                break;
            }
        }
    }
    if (!hasRenderable)
        return;

    _device->setCullFace(false);

    if (_shader) {
        _shader->use();
        _shader->setBool("uRoundPoint", false);
        for (auto& [path, batch] : _lineBatches) {
            if (batch.hidden || batch.vertices.empty() || !batch.vao)
                continue;
            _device->setLineWidth(batch.width);
            batch.vao->bind();
            _device->drawLines(batch.vertices.size());
            batch.vao->unbind();
        }
        _device->setLineWidth(1.0f);

        _shader->setBool("uRoundPoint", true);
        for (auto& [path, batch] : _pointBatches) {
            if (batch.hidden || batch.vertices.empty() || !batch.vao)
                continue;
            batch.vao->bind();
            _device->drawPoints(batch.vertices.size());
            batch.vao->unbind();
        }
    }

    _device->setCullFace(true);
}

} // namespace KE

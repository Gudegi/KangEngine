#ifndef _DEBUG_RENDERER_HPP_
#define _DEBUG_RENDERER_HPP_

// Non-mesh debug overlay renderer for transient lines, points, and axes.
#include "engine/graphics/backend/base/graphics_device.hpp"
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace KE {

class DebugRenderer {
  private:
    struct LineVertex {
        glm::vec3 position;
        glm::vec4 color;
    };

    struct PointVertex {
        glm::vec3 position;
        glm::vec4 color;
        float size;
    };

    struct LineBatch {
        std::unique_ptr<Backend::VertexArray> vao;
        std::unique_ptr<Backend::Buffer> vertexBuffer;
        std::vector<LineVertex> vertices;
        size_t allocatedVertices = 0;
        float width = 1.0f;
        bool hidden = false;
    };

    struct PointBatch {
        std::unique_ptr<Backend::VertexArray> vao;
        std::unique_ptr<Backend::Buffer> vertexBuffer;
        std::vector<PointVertex> vertices;
        size_t allocatedVertices = 0;
        bool hidden = false;
    };

    Backend::GraphicsDevice* _device = nullptr;
    Backend::Shader* _shader = nullptr;
    std::unique_ptr<Backend::Shader> _ownedShader;
    std::map<std::string, LineBatch> _lineBatches;
    std::map<std::string, PointBatch> _pointBatches;

    void ensureLineBatchGpu(LineBatch& batch);
    void ensurePointBatchGpu(PointBatch& batch);

  public:
    DebugRenderer() = default;
    DebugRenderer(DebugRenderer&&) = default;
    DebugRenderer& operator=(DebugRenderer&&) = default;
    DebugRenderer(const DebugRenderer&) = delete;
    DebugRenderer& operator=(const DebugRenderer&) = delete;

    void init(Backend::GraphicsDevice* device);
    void logLines(const std::string& path, const std::vector<glm::vec3>& starts,
                  const std::vector<glm::vec3>& ends,
                  const std::vector<glm::vec4>& colors = {}, float width = 1.0f,
                  bool hidden = false);
    void logAxes(const std::string& path, const glm::mat4& transform,
                 float length = 1.0f, float width = 1.0f, bool hidden = false);
    void logAxes(const std::string& path, const glm::vec3& origin,
                 const glm::vec3& xAxis, const glm::vec3& yAxis,
                 const glm::vec3& zAxis, float length = 1.0f,
                 float width = 1.0f, bool hidden = false);
    void clearLines(const std::string& path);
    void logPoints(const std::string& path,
                   const std::vector<glm::vec3>& points,
                   const std::vector<glm::vec4>& colors = {}, float size = 6.0f,
                   bool hidden = false);
    void clearPoints(const std::string& path);
    void render();
};

} // namespace KE

#endif

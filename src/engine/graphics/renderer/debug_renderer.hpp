#ifndef _DEBUG_RENDERER_HPP_
#define _DEBUG_RENDERER_HPP_

// Non-mesh debug overlay pass owner for transient lines, points, and axes.
// Resource updates happen while authoring batches; record() only emits
// commands.
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/renderer/render_backend/render_debug_draw.hpp"
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace KE {

class ShaderLibrary;

class DebugRenderer : public RenderDebugDraw {
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
        std::unique_ptr<Backend::Buffer> vertexBuffer;
        std::vector<LineVertex> vertices;
        size_t allocatedVertices = 0;
        float width = 1.0f;
        bool hidden = false;
        bool overlay = false;
    };

    struct PointBatch {
        std::unique_ptr<Backend::Buffer> vertexBuffer;
        std::vector<PointVertex> vertices;
        size_t allocatedVertices = 0;
        bool hidden = false;
        bool overlay = false;
    };

    Backend::GraphicsDevice* _device = nullptr;
    std::map<std::string, LineBatch> _lineBatches;
    std::map<std::string, PointBatch> _pointBatches;
    Backend::BindGroup* _frameBindGroup = nullptr;
    std::array<std::unique_ptr<Backend::BindGroupLayout>, 3>
        _reservedGroupLayouts;
    std::unique_ptr<Backend::PipelineLayout> _pipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _linePipeline;
    std::unique_ptr<Backend::GraphicsPipeline> _lineOverlayPipeline;
    std::unique_ptr<Backend::GraphicsPipeline> _pointPipeline;
    std::unique_ptr<Backend::GraphicsPipeline> _pointOverlayPipeline;

    void setLines(const std::string& path, const std::vector<glm::vec3>& starts,
                  const std::vector<glm::vec3>& ends,
                  const std::vector<glm::vec4>& colors, float width,
                  bool hidden, bool overlay) override;
    void setPoints(const std::string& path,
                   const std::vector<glm::vec3>& points,
                   const std::vector<glm::vec4>& colors, float size,
                   bool hidden, bool overlay) override;

    void ensureLineBatchGpu(LineBatch& batch);
    void ensurePointBatchGpu(PointBatch& batch);

  public:
    DebugRenderer() = default;
    DebugRenderer(DebugRenderer&&) = default;
    DebugRenderer& operator=(DebugRenderer&&) = default;
    DebugRenderer(const DebugRenderer&) = delete;
    DebugRenderer& operator=(const DebugRenderer&) = delete;

    void init(Backend::GraphicsDevice* device,
              Backend::BindGroupLayout* frameGroupLayout = nullptr,
              Backend::BindGroup* frameBindGroup = nullptr,
              ShaderLibrary* shaderLibrary = nullptr);
    void clearLines(const std::string& path) override;
    void clearPoints(const std::string& path) override;
    void setVisible(const std::string& path, bool visible) override;
    void remove(const std::string& path) override;
    void removeAll() override;
    bool hasDraws() const;
    void record(Backend::RenderPassEncoder& pass) const;
};

} // namespace KE

#endif

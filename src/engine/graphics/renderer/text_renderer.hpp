#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "render_backend/render_text_draw.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace KE {

class ShaderLibrary;

// Text pass owner. prepare() updates dirty glyph instances on the render
// thread; record() only emits commands into its caller's overlay render pass.
class TextRenderer : public RenderTextDraw {
  private:
    struct QuadVertex {
        glm::vec2 position;
        glm::vec2 uv;
    };

    struct WorldTextEntry {
        WorldTextDesc desc;
    };

    struct ScreenTextEntry {
        ScreenTextDesc desc;
    };

    struct InstanceBatch {
        std::unique_ptr<Backend::Buffer> instanceBuffer;
        std::vector<GlyphInstance> instances;
        std::size_t allocatedInstances = 0;
    };

    Backend::GraphicsDevice* _device = nullptr;
    FontAtlasData _atlasData;
    std::unique_ptr<Backend::Texture> _atlasTexture;
    std::unique_ptr<Backend::Buffer> _quadVertexBuffer;
    std::unique_ptr<Backend::Buffer> _quadIndexBuffer;
    InstanceBatch _depthBatch;
    InstanceBatch _overlayBatch;
    InstanceBatch _screenBatch;
    std::map<std::string, WorldTextEntry> _worldEntries;
    std::map<std::string, ScreenTextEntry> _screenEntries;
    bool _worldDepthDirty = true;
    bool _worldOverlayDirty = true;
    bool _screenDirty = true;
    int _lastViewportWidth = 0;
    int _lastViewportHeight = 0;
    Backend::BindGroup* _frameBindGroup = nullptr;
    std::unique_ptr<Backend::BindGroupLayout> _passGroupLayout;
    std::unique_ptr<Backend::BindGroupLayout> _reservedObjectGroupLayout;
    std::unique_ptr<Backend::BindGroupLayout> _atlasGroupLayout;
    std::unique_ptr<Backend::PipelineLayout> _rhiPipelineLayout;
    std::unique_ptr<Backend::GraphicsPipeline> _worldDepthPipeline;
    std::unique_ptr<Backend::GraphicsPipeline> _worldOverlayPipeline;
    std::unique_ptr<Backend::GraphicsPipeline> _screenPipeline;
    std::unique_ptr<Backend::Buffer> _worldPassParams;
    std::unique_ptr<Backend::Buffer> _screenPassParams;
    std::unique_ptr<Backend::BindGroup> _worldPassBindGroup;
    std::unique_ptr<Backend::BindGroup> _screenPassBindGroup;
    std::unique_ptr<Backend::TextureView> _atlasView;
    std::unique_ptr<Backend::Sampler> _atlasSampler;
    std::unique_ptr<Backend::BindGroup> _atlasBindGroup;

    void createGPUResources();
    void ensureBatchCapacity(InstanceBatch& batch, std::size_t count);
    void rebuildDirtyInstances(int viewportWidth, int viewportHeight);

  public:
    TextRenderer() = default;
    TextRenderer(TextRenderer&&) = default;
    TextRenderer& operator=(TextRenderer&&) = default;
    TextRenderer(const TextRenderer&) = delete;
    TextRenderer& operator=(const TextRenderer&) = delete;

    void init(Backend::GraphicsDevice* device, FontAtlasData atlasData,
              Backend::BindGroupLayout* frameGroupLayout = nullptr,
              Backend::BindGroup* frameBindGroup = nullptr,
              ShaderLibrary* shaderLibrary = nullptr);
    void setWorldText(const std::string& path,
                      const WorldTextDesc& desc) override;
    void setWorldString(const std::string& path, std::string text) override;
    void setWorldPosition(const std::string& path,
                          const glm::vec3& position) override;
    void setWorldHidden(const std::string& path, bool hidden) override;
    void removeWorldText(const std::string& path) override;
    void clearWorldText() override;
    void setScreenText(const std::string& path,
                       const ScreenTextDesc& desc) override;
    void setScreenString(const std::string& path, std::string text) override;
    void setScreenPosition(const std::string& path,
                           const glm::vec2& position) override;
    void setScreenHidden(const std::string& path, bool hidden) override;
    void removeScreenText(const std::string& path) override;
    void clearScreenText() override;
    void prepare(int viewportWidth, int viewportHeight);
    bool hasDraws() const;
    void record(Backend::RenderPassEncoder& pass) const;
};

} // namespace KE

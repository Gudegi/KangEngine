#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"
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

enum class TextAlignment {
    Left,
    Center,
    Right,
};

enum class TextDepthMode {
    DepthTested,
    Overlay,
};

enum class ScreenAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
};

struct GlyphInfo {
    glm::vec2 uvMin{0.0f};
    glm::vec2 uvMax{0.0f};
    glm::vec2 size{0.0f};
    glm::vec2 bearing{0.0f};
    float advance = 0.0f;
};

// Backend-independent bitmap and metrics. The GPU texture is owned by
// TextRenderer so the same atlas data can later be uploaded by another backend.
class FontAtlasData {
  public:
    static constexpr char32_t FirstCodepoint = 32;
    static constexpr char32_t LastCodepoint = 126;
    static constexpr std::size_t GlyphCount =
        LastCodepoint - FirstCodepoint + 1;

  private:
    int _width = 0;
    int _height = 0;
    float _rasterSize = 0.0f;
    std::vector<uint8_t> _pixels;
    std::array<GlyphInfo, GlyphCount> _glyphs{};

  public:
    static FontAtlasData loadAscii(const std::string& fontPath,
                                   float rasterSize = 48.0f,
                                   int atlasWidth = 512, int atlasHeight = 512);

    const GlyphInfo* findGlyph(char32_t codepoint) const;
    int width() const { return _width; }
    int height() const { return _height; }
    float rasterSize() const { return _rasterSize; }
    const std::vector<uint8_t>& pixels() const { return _pixels; }
};

struct WorldTextDesc {
    std::string text;
    glm::vec3 position{0.0f};
    glm::vec4 color{1.0f};
    float pixelSize = 18.0f;
    TextAlignment alignment = TextAlignment::Center;
    TextDepthMode depthMode = TextDepthMode::DepthTested;
    bool hidden = false;
};

struct ScreenTextDesc {
    std::string text;
    glm::vec2 position{0.0f};
    glm::vec4 color{1.0f};
    float pixelSize = 18.0f;
    TextAlignment alignment = TextAlignment::Left;
    ScreenAnchor anchor = ScreenAnchor::TopLeft;
    bool hidden = false;
};

class TextRenderer {
  private:
    struct QuadVertex {
        glm::vec2 position;
        glm::vec2 uv;
    };

    struct GlyphInstance {
        glm::vec3 origin;
        glm::vec2 offset;
        glm::vec2 size;
        glm::vec4 uvRect;
        glm::vec4 color;
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
    void appendWorldEntry(const WorldTextDesc& desc,
                          std::vector<GlyphInstance>& instances) const;
    void appendScreenEntry(const ScreenTextDesc& desc, int viewportWidth,
                           int viewportHeight,
                           std::vector<GlyphInstance>& instances) const;
    void rebuildDirtyInstances(int viewportWidth, int viewportHeight);

  public:
    TextRenderer() = default;
    TextRenderer(TextRenderer&&) = default;
    TextRenderer& operator=(TextRenderer&&) = default;
    TextRenderer(const TextRenderer&) = delete;
    TextRenderer& operator=(const TextRenderer&) = delete;

    void init(Backend::GraphicsDevice* device, FontAtlasData atlasData,
              Backend::BindGroupLayout* frameGroupLayout = nullptr,
              Backend::BindGroup* frameBindGroup = nullptr);
    void setWorldText(const std::string& path, const WorldTextDesc& desc);
    void setWorldString(const std::string& path, std::string text);
    void setWorldPosition(const std::string& path, const glm::vec3& position);
    void setWorldHidden(const std::string& path, bool hidden);
    void removeWorldText(const std::string& path);
    void clearWorldText();
    void setScreenText(const std::string& path, const ScreenTextDesc& desc);
    void setScreenString(const std::string& path, std::string text);
    void setScreenPosition(const std::string& path, const glm::vec2& position);
    void setScreenHidden(const std::string& path, bool hidden);
    void removeScreenText(const std::string& path);
    void clearScreenText();
    void render(Backend::RenderTarget* target, int viewportWidth,
                int viewportHeight);
};

} // namespace KE

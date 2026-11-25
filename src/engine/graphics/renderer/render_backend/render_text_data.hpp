#pragma once

// Font pixels and text placement used by Native and external renderers.

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
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

struct GlyphInstance {
    glm::vec3 origin;
    glm::vec2 offset;
    glm::vec2 size;
    glm::vec4 uvRect;
    glm::vec4 color;
};

struct GlyphInfo {
    glm::vec2 uvMin{0.0f};
    glm::vec2 uvMax{0.0f};
    glm::vec2 size{0.0f};
    glm::vec2 bearing{0.0f};
    float advance = 0.0f;
};

// CPU font bitmap and glyph layout shared by render backends.
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

    void appendWorldText(const WorldTextDesc& desc,
                         std::vector<GlyphInstance>& instances) const;
    void appendScreenText(const ScreenTextDesc& desc, int viewportWidth,
                          int viewportHeight,
                          std::vector<GlyphInstance>& instances) const;
    const GlyphInfo* findGlyph(char32_t codepoint) const;
    int width() const { return _width; }
    int height() const { return _height; }
    float rasterSize() const { return _rasterSize; }
    const std::vector<uint8_t>& pixels() const { return _pixels; }
};

} // namespace KE

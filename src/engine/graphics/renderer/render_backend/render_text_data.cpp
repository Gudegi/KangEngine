#include "render_text_data.hpp"

#include <fstream>
#include <iterator>
#include <stdexcept>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

namespace KE {
namespace {

std::vector<uint8_t> readBinaryFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("TextRenderer could not open font: " + path);
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

char32_t decodeAsciiOrFallback(unsigned char value) {
    return value >= FontAtlasData::FirstCodepoint &&
                   value <= FontAtlasData::LastCodepoint
               ? static_cast<char32_t>(value)
               : U'?';
}

} // namespace

FontAtlasData FontAtlasData::loadAscii(const std::string& fontPath,
                                       float rasterSize, int atlasWidth,
                                       int atlasHeight) {
    if (rasterSize <= 0.0f || atlasWidth <= 0 || atlasHeight <= 0)
        throw std::invalid_argument(
            "FontAtlasData requires positive raster and atlas sizes");

    FontAtlasData result;
    result._width = atlasWidth;
    result._height = atlasHeight;
    result._rasterSize = rasterSize;
    result._pixels.resize(static_cast<std::size_t>(atlasWidth) * atlasHeight);

    const std::vector<uint8_t> fontBytes = readBinaryFile(fontPath);
    std::array<stbtt_bakedchar, GlyphCount> baked{};
    const int bakeResult = stbtt_BakeFontBitmap(
        fontBytes.data(), 0, rasterSize, result._pixels.data(), atlasWidth,
        atlasHeight, static_cast<int>(FirstCodepoint),
        static_cast<int>(GlyphCount), baked.data());
    if (bakeResult <= 0)
        throw std::runtime_error(
            "FontAtlasData atlas is too small for the requested font size");

    for (std::size_t i = 0; i < GlyphCount; ++i) {
        const stbtt_bakedchar& source = baked[i];
        GlyphInfo& glyph = result._glyphs[i];
        glyph.uvMin = {static_cast<float>(source.x0) / atlasWidth,
                       static_cast<float>(source.y0) / atlasHeight};
        glyph.uvMax = {static_cast<float>(source.x1) / atlasWidth,
                       static_cast<float>(source.y1) / atlasHeight};
        glyph.size = {static_cast<float>(source.x1 - source.x0),
                      static_cast<float>(source.y1 - source.y0)};
        glyph.bearing = {source.xoff, source.yoff};
        glyph.advance = source.xadvance;
    }
    return result;
}

const GlyphInfo* FontAtlasData::findGlyph(char32_t codepoint) const {
    if (codepoint < FirstCodepoint || codepoint > LastCodepoint)
        return nullptr;
    return &_glyphs[static_cast<std::size_t>(codepoint - FirstCodepoint)];
}

void FontAtlasData::appendWorldText(
    const WorldTextDesc& desc, std::vector<GlyphInstance>& instances) const {
    if (desc.hidden || desc.text.empty() || desc.pixelSize <= 0.0f)
        return;

    const float scale = desc.pixelSize / rasterSize();
    float totalAdvance = 0.0f;
    for (unsigned char byte : desc.text) {
        if (byte == '\n')
            continue;
        const GlyphInfo* glyph = findGlyph(decodeAsciiOrFallback(byte));
        if (glyph)
            totalAdvance += glyph->advance * scale;
    }

    float penX = 0.0f;
    if (desc.alignment == TextAlignment::Center)
        penX = -0.5f * totalAdvance;
    else if (desc.alignment == TextAlignment::Right)
        penX = -totalAdvance;

    for (unsigned char byte : desc.text) {
        if (byte == '\n')
            continue;
        const GlyphInfo* glyph = findGlyph(decodeAsciiOrFallback(byte));
        if (!glyph)
            continue;

        const glm::vec2 size = glyph->size * scale;
        const float top = -glyph->bearing.y * scale;
        const glm::vec2 offset{
            penX + glyph->bearing.x * scale,
            top - size.y,
        };
        if (size.x > 0.0f && size.y > 0.0f) {
            instances.push_back({desc.position,
                                 offset,
                                 size,
                                 {glyph->uvMin.x, glyph->uvMin.y,
                                  glyph->uvMax.x, glyph->uvMax.y},
                                 desc.color});
        }
        penX += glyph->advance * scale;
    }
}

void FontAtlasData::appendScreenText(
    const ScreenTextDesc& desc, int viewportWidth, int viewportHeight,
    std::vector<GlyphInstance>& instances) const {
    if (desc.hidden || desc.text.empty() || desc.pixelSize <= 0.0f)
        return;

    const float scale = desc.pixelSize / rasterSize();
    float totalAdvance = 0.0f;
    for (unsigned char byte : desc.text) {
        if (byte == '\n')
            continue;
        const GlyphInfo* glyph = findGlyph(decodeAsciiOrFallback(byte));
        if (glyph)
            totalAdvance += glyph->advance * scale;
    }

    glm::vec2 origin = desc.position;
    switch (desc.anchor) {
    case ScreenAnchor::TopCenter:
        origin.x += static_cast<float>(viewportWidth) * 0.5f;
        origin.y += desc.pixelSize;
        break;
    case ScreenAnchor::TopRight:
        origin.x = static_cast<float>(viewportWidth) - desc.position.x;
        origin.y += desc.pixelSize;
        break;
    case ScreenAnchor::CenterLeft:
        origin.y += static_cast<float>(viewportHeight) * 0.5f;
        break;
    case ScreenAnchor::Center:
        origin += glm::vec2(viewportWidth, viewportHeight) * 0.5f;
        break;
    case ScreenAnchor::CenterRight:
        origin.x = static_cast<float>(viewportWidth) - desc.position.x;
        origin.y += static_cast<float>(viewportHeight) * 0.5f;
        break;
    case ScreenAnchor::BottomLeft:
        origin.y = static_cast<float>(viewportHeight) - desc.position.y;
        break;
    case ScreenAnchor::BottomCenter:
        origin.x += static_cast<float>(viewportWidth) * 0.5f;
        origin.y = static_cast<float>(viewportHeight) - desc.position.y;
        break;
    case ScreenAnchor::BottomRight:
        origin = {static_cast<float>(viewportWidth) - desc.position.x,
                  static_cast<float>(viewportHeight) - desc.position.y};
        break;
    case ScreenAnchor::TopLeft:
        origin.y += desc.pixelSize;
        break;
    }

    float penX = 0.0f;
    if (desc.alignment == TextAlignment::Center)
        penX = -0.5f * totalAdvance;
    else if (desc.alignment == TextAlignment::Right)
        penX = -totalAdvance;

    for (unsigned char byte : desc.text) {
        if (byte == '\n')
            continue;
        const GlyphInfo* glyph = findGlyph(decodeAsciiOrFallback(byte));
        if (!glyph)
            continue;

        const glm::vec2 size = glyph->size * scale;
        const glm::vec2 offset{
            penX + glyph->bearing.x * scale,
            glyph->bearing.y * scale,
        };
        if (size.x > 0.0f && size.y > 0.0f) {
            instances.push_back({{origin.x, origin.y, 0.0f},
                                 offset,
                                 size,
                                 {glyph->uvMin.x, glyph->uvMin.y,
                                  glyph->uvMax.x, glyph->uvMax.y},
                                 desc.color});
        }
        penX += glyph->advance * scale;
    }
}

} // namespace KE

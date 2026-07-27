#include "text_renderer.hpp"
#include "utils/asset_path.hpp"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

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

std::size_t growCapacity(std::size_t requested) {
    return std::max<std::size_t>(requested * 2, 64);
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

void TextRenderer::init(Backend::GraphicsDevice* device,
                        FontAtlasData atlasData) {
    _device = device;
    _atlasData = std::move(atlasData);
    if (!_device)
        return;
    createGPUResources();
}

void TextRenderer::createGPUResources() {
    static constexpr QuadVertex vertices[] = {
        {{0.0f, 0.0f}, {0.0f, 1.0f}},
        {{1.0f, 0.0f}, {1.0f, 1.0f}},
        {{1.0f, 1.0f}, {1.0f, 0.0f}},
        {{0.0f, 1.0f}, {0.0f, 0.0f}},
    };
    static constexpr uint32_t indices[] = {0, 1, 2, 0, 2, 3};

    _quadVertexBuffer = _device->createBuffer(Backend::BufferType::Vertex,
                                              sizeof(vertices), vertices);
    _quadIndexBuffer = _device->createBuffer(Backend::BufferType::Index,
                                             sizeof(indices), indices);

    Backend::TextureDesc textureDesc;
    textureDesc.width = _atlasData.width();
    textureDesc.height = _atlasData.height();
    textureDesc.channels = 1;
    textureDesc.data = _atlasData.pixels().data();
    textureDesc.name = "Text Font Atlas";
    Backend::SamplerDesc sampler;
    sampler.wrapU = Backend::TextureWrap::ClampToEdge;
    sampler.wrapV = Backend::TextureWrap::ClampToEdge;
    sampler.minFilter = Backend::TextureFilter::Linear;
    sampler.magFilter = Backend::TextureFilter::Linear;
    _atlasTexture = _device->createTexture(textureDesc, sampler);
    _shader =
        _device->createShaderFromFile(KE::getAssetPath("shaders/text.vs"),
                                      KE::getAssetPath("shaders/text.fs"));
    _shader->setUniformBlockBinding("cameraUBO", 0);
}

void TextRenderer::ensureBatchCapacity(InstanceBatch& batch,
                                       std::size_t count) {
    if (count <= batch.allocatedInstances && batch.vao && batch.instanceBuffer)
        return;

    batch.allocatedInstances = growCapacity(count);
    batch.instanceBuffer =
        _device->createBuffer(Backend::BufferType::DynamicVertex,
                              batch.allocatedInstances * sizeof(GlyphInstance));
    batch.vao = _device->createVertexArray();
    batch.vao->bind();
    batch.vao->setVertexBuffer(_quadVertexBuffer.get());
    batch.vao->setVertexAttribute({0, 2, Backend::VertexAttributeType::Float,
                                   false, sizeof(QuadVertex),
                                   offsetof(QuadVertex, position), 0});
    batch.vao->setVertexAttribute({1, 2, Backend::VertexAttributeType::Float,
                                   false, sizeof(QuadVertex),
                                   offsetof(QuadVertex, uv), 0});
    batch.vao->setVertexBuffer(batch.instanceBuffer.get());
    batch.vao->setVertexAttribute({2, 3, Backend::VertexAttributeType::Float,
                                   false, sizeof(GlyphInstance),
                                   offsetof(GlyphInstance, origin), 1});
    batch.vao->setVertexAttribute({3, 2, Backend::VertexAttributeType::Float,
                                   false, sizeof(GlyphInstance),
                                   offsetof(GlyphInstance, offset), 1});
    batch.vao->setVertexAttribute({4, 2, Backend::VertexAttributeType::Float,
                                   false, sizeof(GlyphInstance),
                                   offsetof(GlyphInstance, size), 1});
    batch.vao->setVertexAttribute({5, 4, Backend::VertexAttributeType::Float,
                                   false, sizeof(GlyphInstance),
                                   offsetof(GlyphInstance, uvRect), 1});
    batch.vao->setVertexAttribute({6, 4, Backend::VertexAttributeType::Float,
                                   false, sizeof(GlyphInstance),
                                   offsetof(GlyphInstance, color), 1});
    batch.vao->setIndexBuffer(_quadIndexBuffer.get());
    batch.vao->unbind();
}

void TextRenderer::appendWorldEntry(
    const WorldTextDesc& desc, std::vector<GlyphInstance>& instances) const {
    if (desc.hidden || desc.text.empty() || desc.pixelSize <= 0.0f)
        return;

    const float scale = desc.pixelSize / _atlasData.rasterSize();
    float totalAdvance = 0.0f;
    for (unsigned char byte : desc.text) {
        if (byte == '\n')
            continue;
        const GlyphInfo* glyph =
            _atlasData.findGlyph(decodeAsciiOrFallback(byte));
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
        const GlyphInfo* glyph =
            _atlasData.findGlyph(decodeAsciiOrFallback(byte));
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

void TextRenderer::appendScreenEntry(
    const ScreenTextDesc& desc, int viewportWidth, int viewportHeight,
    std::vector<GlyphInstance>& instances) const {
    if (desc.hidden || desc.text.empty() || desc.pixelSize <= 0.0f)
        return;

    const float scale = desc.pixelSize / _atlasData.rasterSize();
    float totalAdvance = 0.0f;
    for (unsigned char byte : desc.text) {
        if (byte == '\n')
            continue;
        const GlyphInfo* glyph =
            _atlasData.findGlyph(decodeAsciiOrFallback(byte));
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
        const GlyphInfo* glyph =
            _atlasData.findGlyph(decodeAsciiOrFallback(byte));
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

void TextRenderer::rebuildDirtyInstances(int viewportWidth,
                                         int viewportHeight) {
    if (_worldDepthDirty) {
        _depthBatch.instances.clear();
        for (const auto& [path, entry] : _worldEntries) {
            if (entry.desc.depthMode == TextDepthMode::DepthTested)
                appendWorldEntry(entry.desc, _depthBatch.instances);
        }
    }
    if (_worldOverlayDirty) {
        _overlayBatch.instances.clear();
        for (const auto& [path, entry] : _worldEntries) {
            if (entry.desc.depthMode == TextDepthMode::Overlay)
                appendWorldEntry(entry.desc, _overlayBatch.instances);
        }
    }
    if (_screenDirty) {
        _screenBatch.instances.clear();
        for (const auto& [path, entry] : _screenEntries)
            appendScreenEntry(entry.desc, viewportWidth, viewportHeight,
                              _screenBatch.instances);
    }

    const std::array<std::pair<InstanceBatch*, bool>, 3> batches{{
        {&_depthBatch, _worldDepthDirty},
        {&_overlayBatch, _worldOverlayDirty},
        {&_screenBatch, _screenDirty},
    }};
    for (const auto& [batch, dirty] : batches) {
        if (!dirty)
            continue;
        if (batch->instances.empty())
            continue;
        ensureBatchCapacity(*batch, batch->instances.size());
        batch->instanceBuffer->setData(batch->instances.data(),
                                       batch->instances.size() *
                                           sizeof(GlyphInstance));
    }
    _lastViewportWidth = viewportWidth;
    _lastViewportHeight = viewportHeight;
    _worldDepthDirty = false;
    _worldOverlayDirty = false;
    _screenDirty = false;
}

void TextRenderer::renderBatch(InstanceBatch& batch, bool depthTest,
                               int viewportWidth, int viewportHeight,
                               bool screenSpace) {
    if (batch.instances.empty() || !batch.vao)
        return;

    _device->setDepthTest(depthTest);
    _shader->setVec2("uViewportSize", static_cast<float>(viewportWidth),
                     static_cast<float>(viewportHeight));
    _shader->setBool("uScreenSpace", screenSpace);
    batch.vao->bind();
    _device->drawIndexedInstanced(6, batch.instances.size());
    batch.vao->unbind();
}

void TextRenderer::render(int viewportWidth, int viewportHeight) {
    if (!_device || !_shader || !_atlasTexture || viewportWidth <= 0 ||
        viewportHeight <= 0)
        return;
    if (viewportWidth != _lastViewportWidth ||
        viewportHeight != _lastViewportHeight)
        _screenDirty = true;
    if (_worldDepthDirty || _worldOverlayDirty || _screenDirty)
        rebuildDirtyInstances(viewportWidth, viewportHeight);
    if (_depthBatch.instances.empty() && _overlayBatch.instances.empty() &&
        _screenBatch.instances.empty())
        return;

    _device->setCullFace(false);
    _device->setDepthWrite(false);
    _device->setBlend(true);
    _device->setBlendFunc(Backend::BlendFactor::SrcAlpha,
                          Backend::BlendFactor::OneMinusSrcAlpha);
    _shader->use();
    _shader->setInt("uFontAtlas", 0);
    _atlasTexture->bind(0);

    renderBatch(_depthBatch, true, viewportWidth, viewportHeight, false);
    renderBatch(_overlayBatch, false, viewportWidth, viewportHeight, false);
    renderBatch(_screenBatch, false, viewportWidth, viewportHeight, true);

    _atlasTexture->unbind();
    _device->setBlend(false);
    _device->setDepthWrite(true);
    _device->setDepthTest(true);
    _device->setCullFace(true);
}

void TextRenderer::setWorldText(const std::string& path,
                                const WorldTextDesc& desc) {
    if (path.empty())
        return;
    auto it = _worldEntries.find(path);
    if (it != _worldEntries.end()) {
        const WorldTextDesc& current = it->second.desc;
        if (current.text == desc.text && current.position == desc.position &&
            current.color == desc.color &&
            current.pixelSize == desc.pixelSize &&
            current.alignment == desc.alignment &&
            current.depthMode == desc.depthMode &&
            current.hidden == desc.hidden)
            return;
        if (current.depthMode == TextDepthMode::DepthTested)
            _worldDepthDirty = true;
        else
            _worldOverlayDirty = true;
    }
    _worldEntries[path].desc = desc;
    if (desc.depthMode == TextDepthMode::DepthTested)
        _worldDepthDirty = true;
    else
        _worldOverlayDirty = true;
}

void TextRenderer::setWorldString(const std::string& path, std::string text) {
    auto it = _worldEntries.find(path);
    if (it == _worldEntries.end())
        return;
    if (it->second.desc.text == text)
        return;
    it->second.desc.text = std::move(text);
    if (it->second.desc.depthMode == TextDepthMode::DepthTested)
        _worldDepthDirty = true;
    else
        _worldOverlayDirty = true;
}

void TextRenderer::setWorldPosition(const std::string& path,
                                    const glm::vec3& position) {
    auto it = _worldEntries.find(path);
    if (it == _worldEntries.end())
        return;
    if (it->second.desc.position == position)
        return;
    it->second.desc.position = position;
    if (it->second.desc.depthMode == TextDepthMode::DepthTested)
        _worldDepthDirty = true;
    else
        _worldOverlayDirty = true;
}

void TextRenderer::setWorldHidden(const std::string& path, bool hidden) {
    auto it = _worldEntries.find(path);
    if (it == _worldEntries.end())
        return;
    if (it->second.desc.hidden == hidden)
        return;
    it->second.desc.hidden = hidden;
    if (it->second.desc.depthMode == TextDepthMode::DepthTested)
        _worldDepthDirty = true;
    else
        _worldOverlayDirty = true;
}

void TextRenderer::removeWorldText(const std::string& path) {
    auto it = _worldEntries.find(path);
    if (it == _worldEntries.end())
        return;
    if (it->second.desc.depthMode == TextDepthMode::DepthTested)
        _worldDepthDirty = true;
    else
        _worldOverlayDirty = true;
    _worldEntries.erase(it);
}

void TextRenderer::clearWorldText() {
    if (_worldEntries.empty())
        return;
    _worldEntries.clear();
    _worldDepthDirty = true;
    _worldOverlayDirty = true;
}

void TextRenderer::setScreenText(const std::string& path,
                                 const ScreenTextDesc& desc) {
    if (path.empty())
        return;
    auto it = _screenEntries.find(path);
    if (it != _screenEntries.end()) {
        const ScreenTextDesc& current = it->second.desc;
        if (current.text == desc.text && current.position == desc.position &&
            current.color == desc.color &&
            current.pixelSize == desc.pixelSize &&
            current.alignment == desc.alignment &&
            current.anchor == desc.anchor && current.hidden == desc.hidden)
            return;
    }
    _screenEntries[path].desc = desc;
    _screenDirty = true;
}

void TextRenderer::setScreenString(const std::string& path, std::string text) {
    auto it = _screenEntries.find(path);
    if (it == _screenEntries.end())
        return;
    if (it->second.desc.text == text)
        return;
    it->second.desc.text = std::move(text);
    _screenDirty = true;
}

void TextRenderer::setScreenPosition(const std::string& path,
                                     const glm::vec2& position) {
    auto it = _screenEntries.find(path);
    if (it == _screenEntries.end())
        return;
    if (it->second.desc.position == position)
        return;
    it->second.desc.position = position;
    _screenDirty = true;
}

void TextRenderer::setScreenHidden(const std::string& path, bool hidden) {
    auto it = _screenEntries.find(path);
    if (it == _screenEntries.end())
        return;
    if (it->second.desc.hidden == hidden)
        return;
    it->second.desc.hidden = hidden;
    _screenDirty = true;
}

void TextRenderer::removeScreenText(const std::string& path) {
    if (_screenEntries.erase(path) > 0)
        _screenDirty = true;
}

void TextRenderer::clearScreenText() {
    if (_screenEntries.empty())
        return;
    _screenEntries.clear();
    _screenDirty = true;
}

} // namespace KE

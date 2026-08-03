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
                        FontAtlasData atlasData,
                        Backend::BindGroupLayout* frameGroupLayout,
                        Backend::BindGroup* frameBindGroup) {
    _device = device;
    _frameBindGroup = frameBindGroup;
    _atlasData = std::move(atlasData);
    if (!_device)
        return;
    createGPUResources();

    if (!frameGroupLayout || !_frameBindGroup)
        return;
    Backend::BindGroupLayoutDesc passLayoutDesc;
    passLayoutDesc.label = "text_pass_group_layout";
    passLayoutDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                               Backend::ShaderStageVisibility::Vertex}};
    _passGroupLayout = _device->createBindGroupLayout(passLayoutDesc);
    Backend::BindGroupLayoutDesc emptyLayoutDesc;
    emptyLayoutDesc.label = "text_reserved_object_group_layout";
    _reservedObjectGroupLayout =
        _device->createBindGroupLayout(emptyLayoutDesc);
    Backend::BindGroupLayoutDesc atlasLayoutDesc;
    atlasLayoutDesc.label = "text_atlas_group_layout";
    atlasLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {1, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment}};
    _atlasGroupLayout = _device->createBindGroupLayout(atlasLayoutDesc);
    Backend::PipelineLayoutDesc pipelineLayoutDesc;
    pipelineLayoutDesc.label = "text_pipeline_layout";
    pipelineLayoutDesc.bindGroupLayouts = {
        frameGroupLayout, _passGroupLayout.get(),
        _reservedObjectGroupLayout.get(), _atlasGroupLayout.get()};
    _rhiPipelineLayout = _device->createPipelineLayout(pipelineLayoutDesc);

    Backend::BufferDesc paramsDesc;
    paramsDesc.size = sizeof(glm::vec4);
    paramsDesc.usage = Backend::BufferUsage::Uniform |
                       Backend::BufferUsage::CopyDst;
    paramsDesc.label = "text_world_pass_params";
    _worldPassParams = _device->createBuffer(paramsDesc);
    paramsDesc.label = "text_screen_pass_params";
    _screenPassParams = _device->createBuffer(paramsDesc);
    Backend::BindGroupDesc groupDesc;
    groupDesc.layout = _passGroupLayout.get();
    groupDesc.label = "text_world_pass_bind_group";
    groupDesc.entries = {{0, _worldPassParams.get(), 0, sizeof(glm::vec4),
                          nullptr, nullptr}};
    _worldPassBindGroup = _device->createBindGroup(groupDesc);
    groupDesc.label = "text_screen_pass_bind_group";
    groupDesc.entries[0].buffer = _screenPassParams.get();
    _screenPassBindGroup = _device->createBindGroup(groupDesc);

    Backend::TextureViewDesc atlasViewDesc;
    atlasViewDesc.label = "text_atlas_view";
    _atlasView = _device->createTextureView(_atlasTexture.get(), atlasViewDesc);
    Backend::SamplerDesc atlasSamplerDesc;
    atlasSamplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    atlasSamplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    atlasSamplerDesc.minFilter = Backend::TextureFilter::Linear;
    atlasSamplerDesc.magFilter = Backend::TextureFilter::Linear;
    atlasSamplerDesc.label = "text_atlas_sampler";
    _atlasSampler = _device->createSampler(atlasSamplerDesc);
    Backend::BindGroupDesc atlasGroupDesc;
    atlasGroupDesc.layout = _atlasGroupLayout.get();
    atlasGroupDesc.label = "text_atlas_bind_group";
    atlasGroupDesc.entries = {
        {0, nullptr, 0, 0, _atlasView.get(), nullptr},
        {1, nullptr, 0, 0, nullptr, _atlasSampler.get()}};
    _atlasBindGroup = _device->createBindGroup(atlasGroupDesc);

    Backend::VertexBufferLayout quadLayout;
    quadLayout.arrayStride = sizeof(QuadVertex);
    quadLayout.attributes = {
        {Backend::VertexFormat::Float32x2, offsetof(QuadVertex, position), 0},
        {Backend::VertexFormat::Float32x2, offsetof(QuadVertex, uv), 1}};
    Backend::VertexBufferLayout instanceLayout;
    instanceLayout.arrayStride = sizeof(GlyphInstance);
    instanceLayout.stepMode = Backend::VertexStepMode::Instance;
    instanceLayout.attributes = {
        {Backend::VertexFormat::Float32x3, offsetof(GlyphInstance, origin), 2},
        {Backend::VertexFormat::Float32x2, offsetof(GlyphInstance, offset), 3},
        {Backend::VertexFormat::Float32x2, offsetof(GlyphInstance, size), 4},
        {Backend::VertexFormat::Float32x4, offsetof(GlyphInstance, uvRect), 5},
        {Backend::VertexFormat::Float32x4, offsetof(GlyphInstance, color), 6}};
    Backend::BlendState blend;
    blend.color.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    blend.color.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    blend.alpha.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    blend.alpha.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "text_world_depth_pipeline";
    pipelineDesc.shader.name = "text_rhi";
    pipelineDesc.shader.stages = {
        {Backend::loadShaderSource(KE::getAssetPath("shaders/rhi/text.vs")),
         Backend::ShaderType::Vertex, "main"},
        {Backend::loadShaderSource(KE::getAssetPath("shaders/rhi/text.fs")),
         Backend::ShaderType::Fragment, "main"}};
    pipelineDesc.pipelineLayout = _rhiPipelineLayout.get();
    pipelineDesc.vertexBuffers = {quadLayout, instanceLayout};
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.depthStencil = Backend::DepthStencilState{
        Backend::TextureFormat::Depth24Stencil8, false,
        Backend::CompareFunction::Less};
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float, blend}};
    pipelineDesc.sampleCount = 4;
    _worldDepthPipeline = _device->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "text_world_overlay_pipeline";
    pipelineDesc.depthStencil->depthCompare = Backend::CompareFunction::Always;
    _worldOverlayPipeline = _device->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "text_screen_pipeline";
    _screenPipeline = _device->createGraphicsPipeline(pipelineDesc);
}

void TextRenderer::createGPUResources() {
    static constexpr QuadVertex vertices[] = {
        {{0.0f, 0.0f}, {0.0f, 1.0f}},
        {{1.0f, 0.0f}, {1.0f, 1.0f}},
        {{1.0f, 1.0f}, {1.0f, 0.0f}},
        {{0.0f, 1.0f}, {0.0f, 0.0f}},
    };
    static constexpr uint32_t indices[] = {0, 1, 2, 0, 2, 3};

    Backend::BufferDesc bufferDesc;
    bufferDesc.size = sizeof(vertices);
    bufferDesc.usage = Backend::BufferUsage::Vertex;
    bufferDesc.label = "text_quad_vertices";
    _quadVertexBuffer = _device->createBuffer(bufferDesc, vertices);
    bufferDesc.size = sizeof(indices);
    bufferDesc.usage = Backend::BufferUsage::Index;
    bufferDesc.label = "text_quad_indices";
    _quadIndexBuffer = _device->createBuffer(bufferDesc, indices);

    Backend::TextureResourceDesc textureDesc;
    textureDesc.extent = {static_cast<uint32_t>(_atlasData.width()),
                          static_cast<uint32_t>(_atlasData.height()), 1};
    textureDesc.format = Backend::TextureFormat::R8Unorm;
    textureDesc.usage = Backend::TextureUsage::TextureBinding |
                        Backend::TextureUsage::CopyDst;
    textureDesc.label = "text_font_atlas";
    Backend::TextureInitialData initialData;
    initialData.data = _atlasData.pixels().data();
    initialData.size = _atlasData.pixels().size();
    initialData.bytesPerRow = static_cast<size_t>(_atlasData.width());
    _atlasTexture = _device->createTexture(textureDesc, &initialData);
}

void TextRenderer::ensureBatchCapacity(InstanceBatch& batch,
                                       std::size_t count) {
    if (count <= batch.allocatedInstances && batch.instanceBuffer)
        return;

    batch.allocatedInstances = growCapacity(count);
    Backend::BufferDesc desc;
    desc.size = batch.allocatedInstances * sizeof(GlyphInstance);
    desc.usage = Backend::BufferUsage::Vertex |
                 Backend::BufferUsage::CopyDst;
    desc.label = "text_glyph_instances";
    batch.instanceBuffer = _device->createBuffer(desc);
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

void TextRenderer::render(Backend::RenderTarget* target, int viewportWidth,
                          int viewportHeight) {
    if (!target || !_device || !_worldDepthPipeline || !_atlasBindGroup ||
        viewportWidth <= 0 || viewportHeight <= 0)
        return;
    if (viewportWidth != _lastViewportWidth ||
        viewportHeight != _lastViewportHeight)
        _screenDirty = true;
    if (_worldDepthDirty || _worldOverlayDirty || _screenDirty)
        rebuildDirtyInstances(viewportWidth, viewportHeight);
    if (_depthBatch.instances.empty() && _overlayBatch.instances.empty() &&
        _screenBatch.instances.empty())
        return;

    const glm::vec4 worldParams{static_cast<float>(viewportWidth),
                                static_cast<float>(viewportHeight), 0.0f, 0.0f};
    const glm::vec4 screenParams{static_cast<float>(viewportWidth),
                                 static_cast<float>(viewportHeight), 1.0f, 0.0f};
    _worldPassParams->setData(&worldParams, sizeof(worldParams));
    _screenPassParams->setData(&screenParams, sizeof(screenParams));

    auto encoder = _device->createCommandEncoder();
    auto pass = encoder->beginRenderPass(target);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(viewportWidth),
                      static_cast<float>(viewportHeight));
    auto recordBatch = [&](InstanceBatch& batch,
                           Backend::GraphicsPipeline* pipeline,
                           Backend::BindGroup* paramsGroup) {
        if (batch.instances.empty() || !batch.instanceBuffer)
            return;
        pass->setPipeline(pipeline);
        pass->setBindGroup(0, _frameBindGroup);
        pass->setBindGroup(1, paramsGroup);
        pass->setBindGroup(3, _atlasBindGroup.get());
        pass->setVertexBuffer(0, _quadVertexBuffer.get());
        pass->setVertexBuffer(1, batch.instanceBuffer.get());
        pass->setIndexBuffer(_quadIndexBuffer.get(),
                             Backend::IndexFormat::Uint32);
        pass->drawIndexed(6, static_cast<uint32_t>(batch.instances.size()));
    };
    recordBatch(_depthBatch, _worldDepthPipeline.get(),
                _worldPassBindGroup.get());
    recordBatch(_overlayBatch, _worldOverlayPipeline.get(),
                _worldPassBindGroup.get());
    recordBatch(_screenBatch, _screenPipeline.get(),
                _screenPassBindGroup.get());
    pass->end();
    auto commands = encoder->finish();
    _device->submit(*commands);
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

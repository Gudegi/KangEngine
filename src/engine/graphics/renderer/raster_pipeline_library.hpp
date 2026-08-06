#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <tuple>

namespace KE {

// Identifies shader families owned by the raster renderer. This deliberately
// stays raster-specific; compute and ray-tracing pipelines will use their own
// keys and libraries instead of growing this enum into a global pipeline key.
enum class RasterPipelineFamily {
    VertexColor,
    TexturedVertexColor,
    GroundPhong,
    GroundPbr,
    DebugChecker,
    Phong,
    Pbr,
};

// Render-pass compatibility portion of a graphics pipeline key. A pipeline
// compiled for one attachment/sample/topology contract must never be reused by
// another pass, even when both use the same material/shader family.
struct RasterPassSignature {
    static constexpr size_t MaxColorTargets = 8;

    std::array<Backend::TextureFormat, MaxColorTargets> colorFormats{};
    uint8_t colorTargetCount = 0;
    Backend::TextureFormat depthFormat = Backend::TextureFormat::Undefined;
    uint32_t sampleCount = 1;
    Backend::PrimitiveTopology topology =
        Backend::PrimitiveTopology::TriangleList;

    static RasterPassSignature
    fromPipelineDesc(const Backend::GraphicsPipelineDesc& desc);

    bool operator==(const RasterPassSignature& other) const {
        return std::tie(colorFormats, colorTargetCount, depthFormat,
                        sampleCount, topology) ==
               std::tie(other.colorFormats, other.colorTargetCount,
                        other.depthFormat, other.sampleCount, other.topology);
    }
    bool operator<(const RasterPassSignature& other) const {
        return std::tie(colorFormats, colorTargetCount, depthFormat,
                        sampleCount, topology) <
               std::tie(other.colorFormats, other.colorTargetCount,
                        other.depthFormat, other.sampleCount, other.topology);
    }
};

/* examples
  Phong + static + opaque + double-sided
    → forward_phong_double_sided_pipeline

  PBR + skinned + transparent + back-face
    → forward_skinned_pbr_transparent_pipeline

  VertexColor + static + opaque + back-face
    → forward_vertex_color_pipeline
*/
// TODO:: lazy cache
struct RasterPipelineKey {
    RasterPipelineFamily family = RasterPipelineFamily::VertexColor;
    bool skinned = false;
    bool transparent = false;
    bool doubleSided = false;
    RasterPassSignature pass;
    uint64_t shaderGeneration = 0;

    RasterPipelineKey() = default;
    RasterPipelineKey(RasterPipelineFamily familyValue, bool skinnedValue,
                      bool transparentValue, bool doubleSidedValue)
        : family(familyValue), skinned(skinnedValue),
          transparent(transparentValue), doubleSided(doubleSidedValue) {}

    bool operator<(const RasterPipelineKey& other) const {
        return std::tie(family, skinned, transparent, doubleSided, pass,
                        shaderGeneration) <
               std::tie(other.family, other.skinned, other.transparent,
                        other.doubleSided, other.pass, other.shaderGeneration);
    }
};

// Owns immutable raster pipeline variants. Pipelines are registered during
// renderer initialization and only queried while recording frames.
class RasterPipelineLibrary {
  public:
    using Factory = std::function<std::unique_ptr<Backend::GraphicsPipeline>()>;

    void add(RasterPipelineKey key,
             std::unique_ptr<Backend::GraphicsPipeline> pipeline);
    Backend::GraphicsPipeline* find(const RasterPipelineKey& key) const;
    Backend::GraphicsPipeline* getOrCreate(RasterPipelineKey key,
                                           const Factory& factory);
    Backend::GraphicsPipeline* get(RasterPipelineKey key) const;
    void retireBeforeShaderGeneration(uint64_t generation);
    size_t size() const { return _pipelines.size(); }

  private:
    std::map<RasterPipelineKey, std::unique_ptr<Backend::GraphicsPipeline>>
        _pipelines;
};

} // namespace KE

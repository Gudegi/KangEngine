#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"

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
    Checkerboard,
    DebugChecker,
    Phong,
    Pbr,
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

    bool operator<(const RasterPipelineKey& other) const {
        return std::tie(family, skinned, transparent, doubleSided) <
               std::tie(other.family, other.skinned, other.transparent,
                        other.doubleSided);
    }
};

// Owns immutable raster pipeline variants. Pipelines are registered during
// renderer initialization and only queried while recording frames.
class RasterPipelineLibrary {
  public:
    void add(RasterPipelineKey key,
             std::unique_ptr<Backend::GraphicsPipeline> pipeline);
    Backend::GraphicsPipeline* get(RasterPipelineKey key) const;

  private:
    std::map<RasterPipelineKey, std::unique_ptr<Backend::GraphicsPipeline>>
        _pipelines;
};

} // namespace KE
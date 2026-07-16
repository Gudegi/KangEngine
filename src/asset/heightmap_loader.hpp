#pragma once

#include <string>

#include "engine/scene/scene_backend.hpp"
#include "utils/types.hpp"

namespace KE {
namespace Asset {

struct HeightFieldMeshOptions {
    UpAxis upAxis = UpAxis::Y;
    float horizontalScale = 1.0f;
    bool center = true;
};

struct HeightmapTerrainOptions {
    UpAxis upAxis = UpAxis::Y;
    // Distance between adjacent source pixels in world units.
    float horizontalScale = 1.0f;
    // Height range applied to normalized pixel values in [0, 1].
    float heightScale = 64.0f;
    // Height offset added after scaling. Default preserves the previous
    // roughly [-16, 48] terrain range.
    float heightOffset = -16.0f;
    // Sample every Nth source pixel. The final source row/column is always
    // included, so terrain bounds stay stable when downsampling.
    int sampleStride = 1;
};

class HeightmapLoader {
  public:
    HeightmapLoader() = delete;

    static Scene::MeshData
    loadHeightMapTerrain(const std::string& path,
                         HeightmapTerrainOptions options = {});
};

Scene::MeshData heightFieldToMesh(const float* heights, int rows, int cols,
                                  const HeightFieldMeshOptions& options = {});
} // namespace Asset

} // namespace KE

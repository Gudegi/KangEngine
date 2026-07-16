#include "asset/heightmap_loader.hpp"
#include "engine/scene/scene_backend.hpp"
#include "geometry/mesh_utils.hpp"
#include "utils/types.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <stb_image.h>

namespace KE {
namespace Asset {

namespace {
float readHeightValue(const unsigned char* texel, int channels,
                      const HeightmapTerrainOptions& options) {
    if (!texel || channels <= 0)
        return options.heightOffset;
    return (static_cast<float>(texel[0]) / 255.0f) * options.heightScale +
           options.heightOffset;
}
} // namespace

Scene::MeshData heightFieldToMesh(const float* heights, int rows, int cols,
                                  const HeightFieldMeshOptions& options) {
    Scene::MeshData meshData;
    if (!heights || rows < 2 || cols < 2)
        return meshData;

    meshData.vertices.reserve(static_cast<size_t>(rows) *
                              static_cast<size_t>(cols));
    meshData.uvs.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    meshData.indices.reserve(static_cast<size_t>(rows - 1) *
                             static_cast<size_t>(cols - 1) * 6);

    const float originX =
        options.center ? (static_cast<float>(cols - 1) * 0.5f) : 0.0f;
    const float originZ =
        options.center ? (static_cast<float>(rows - 1) * 0.5f) : 0.0f;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const float height = heights[row * cols + col];
            const float x =
                (static_cast<float>(col) - originX) * options.horizontalScale;
            const float z =
                (static_cast<float>(row) - originZ) * options.horizontalScale;

            if (options.upAxis == UpAxis::Z)
                meshData.vertices.emplace_back(x, z, height);
            else
                meshData.vertices.emplace_back(x, height, z);

            meshData.uvs.emplace_back(
                static_cast<float>(col) / static_cast<float>(cols - 1),
                static_cast<float>(row) / static_cast<float>(rows - 1));
        }
    }

    for (int row = 0; row < rows - 1; ++row) {
        for (int col = 0; col < cols - 1; ++col) {
            const unsigned int i0 = static_cast<unsigned int>(row * cols + col);
            const unsigned int i1 = i0 + 1;
            const unsigned int i2 =
                static_cast<unsigned int>((row + 1) * cols + col);
            const unsigned int i3 = i2 + 1;

            meshData.indices.emplace_back(i0);
            meshData.indices.emplace_back(i2);
            meshData.indices.emplace_back(i1);

            meshData.indices.emplace_back(i1);
            meshData.indices.emplace_back(i2);
            meshData.indices.emplace_back(i3);
        }
    }

    meshData.fillMissingAttributes();
    Geometry::computeTangents(meshData);
    return meshData;
}

Scene::MeshData
HeightmapLoader::loadHeightMapTerrain(const std::string& path,
                                      HeightmapTerrainOptions options) {

    Scene::MeshData meshData;
    options.sampleStride = std::max(1, options.sampleStride);

    int imageWidth = 0;
    int imageHeight = 0;
    int channels = 0;
    unsigned char* data =
        stbi_load(path.c_str(), &imageWidth, &imageHeight, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load the Heightmap image '" << path
                  << "': " << stbi_failure_reason() << std::endl;
        return meshData; // right?
    }
    if (imageWidth < 2 || imageHeight < 2) {
        std::cerr << "Heightmap image must be at least 2x2: '" << path << "'"
                  << std::endl;
        stbi_image_free(data);
        return meshData;
    }

    const int sampledWidth = (imageWidth - 2) / options.sampleStride + 2;
    const int sampledHeight = (imageHeight - 2) / options.sampleStride + 2;
    std::vector<float> heights;
    heights.reserve(static_cast<size_t>(sampledWidth) *
                    static_cast<size_t>(sampledHeight));

    // vertices
    for (int sampleRow = 0; sampleRow < sampledHeight; sampleRow++) {
        const int row =
            std::min(sampleRow * options.sampleStride, imageHeight - 1);
        for (int sampleCol = 0; sampleCol < sampledWidth; sampleCol++) {
            // imageWidth = 10, stride = 3 -> samples = 0, 3, 6, 9
            const int col =
                std::min(sampleCol * options.sampleStride, imageWidth - 1);
            // retrieve texel memory address
            unsigned char* texel = data + (row * imageWidth + col) * channels;
            float heightValue = readHeightValue(texel, channels, options);
            heights.push_back(heightValue);
        }
    }
    stbi_image_free(data);

    HeightFieldMeshOptions meshOptions;
    meshOptions.upAxis = options.upAxis;
    meshOptions.horizontalScale =
        options.horizontalScale * static_cast<float>(options.sampleStride);
    meshOptions.center = true;
    return heightFieldToMesh(heights.data(), sampledHeight, sampledWidth,
                             meshOptions);
}

} // namespace Asset
} // namespace KE

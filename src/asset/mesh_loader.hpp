#ifndef _MESH_LOADER_HPP_
#define _MESH_LOADER_HPP_

#include "engine/scene/scene_backend.hpp"

#include <optional>
#include <array>
#include <string>
#include <tiny_obj_loader.h>
#include <vector>

// Support obj and stl.

namespace KE {
namespace Asset {

struct ObjMaterialInfo {
    std::string name;
    std::array<float, 4> ambientColor = {0.0f, 0.0f, 0.0f, 1.0f};
    std::array<float, 4> diffuseColor = {1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 4> specularColor = {0.0f, 0.0f, 0.0f, 1.0f};
    float shininess = 1.0f;
    std::string diffuseTexturePath;
    std::string specularTexturePath;
    std::string alphaTexturePath;
    std::string normalTexturePath;
    bool hasDiffuseTexture = false;
    bool hasSpecularTexture = false;
    bool hasAlphaTexture = false;
    bool hasNormalTexture = false;
};

struct ObjMeshSubsetInfo {
    std::string name;
    int materialIndex = -1;
    Scene::MeshData meshData;
};

struct ObjMeshInfo {
    Scene::MeshData meshData;
    std::vector<ObjMaterialInfo> materials;
    std::vector<ObjMeshSubsetInfo> subsets;
    int primaryMaterialIndex = -1;
};

Scene::MeshData
loadObj(std::string inputFile,
        std::optional<tinyobj::ObjReaderConfig> render_config = std::nullopt);

ObjMeshInfo loadObjWithMaterials(
    std::string inputFile,
    std::optional<tinyobj::ObjReaderConfig> render_config = std::nullopt);

/// @brief Load a mesh from an STL file (supports both ASCII and Binary)
/// @param path Path to the .stl file
/// @return MeshData with vertices, normals, and indices (UVs are empty)
Scene::MeshData loadStl(const std::string& path);

/// Rebuild indices by sharing vertices with identical position/normal/uv.
/// Useful after loaders expand face corners for face-varying attributes.
Scene::MeshData deduplicateMeshData(const Scene::MeshData& meshData);

} // namespace Asset
} // namespace KE

#endif

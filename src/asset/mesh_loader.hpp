#ifndef _MESH_LOADER_HPP_
#define _MESH_LOADER_HPP_

#include "engine/scene/scene_backend.hpp"

#include <optional>
#include <string>
#include <tiny_obj_loader.h>

// Support obj and stl.

namespace KE {
namespace Asset {

Scene::MeshData
loadObj(std::string inputFile,
        std::optional<tinyobj::ObjReaderConfig> render_config = std::nullopt);

/// @brief Load a mesh from an STL file (supports both ASCII and Binary)
/// @param path Path to the .stl file
/// @return MeshData with vertices, normals, and indices (UVs are empty)
Scene::MeshData loadStl(const std::string& path);

} // namespace Asset
} // namespace KE

#endif

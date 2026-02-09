#ifndef __LOAD_UTILS_HPP__
#define __LOAD_UTILS_HPP__

#include <optional>
#include <tiny_obj_loader.h>
#include <string>
#include "scene/scene_backend.hpp"

namespace KE {

KE::Scene::MeshData
loadObj(std::string inputFile,
        std::optional<tinyobj::ObjReaderConfig> render_config = std::nullopt);

/// @brief Load a mesh from an STL file (supports both ASCII and Binary)
/// @param path Path to the .stl file
/// @return MeshData with vertices, normals, and indices (UVs are empty)
KE::Scene::MeshData loadStl(const std::string& path);

} // namespace KE

#endif
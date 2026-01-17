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

} // namespace KE

#endif
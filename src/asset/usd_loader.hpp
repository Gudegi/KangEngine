#ifndef _USD_LOADER_HPP_
#define _USD_LOADER_HPP_

#include "asset/import_diagnostics.hpp"
#include "engine/scene/scene_backend.hpp"

#include <string>
#include <vector>

namespace KE {
namespace Asset {

struct USDMeshInfo {
    std::string primPath;
    std::string name;
    std::string materialPath;
    std::string diffuseTexturePath;
    std::string normalTexturePath;
    Scene::MeshData meshData;
};

struct USDImportResult {
    std::vector<USDMeshInfo> meshes;
    ImportDiagnostics diagnostics;
};

class USDLoader {
  public:
    USDLoader() = delete;

    static USDImportResult parse(const std::string& usdPath,
                                 float scale = 1.0f);

    static std::vector<USDMeshInfo> loadMeshes(const std::string& usdPath,
                                               float scale = 1.0f);
};

} // namespace Asset
} // namespace KE

#endif

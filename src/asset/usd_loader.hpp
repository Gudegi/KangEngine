#ifndef _USD_LOADER_HPP_
#define _USD_LOADER_HPP_

#include "asset/import_diagnostics.hpp"
#include "asset/articulation_desc.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/types/light.hpp"
#include "utils/types.hpp"

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
    std::string metallicRoughnessTexturePath;
    std::string metallicTexturePath;
    std::string roughnessTexturePath;
    std::string emissiveTexturePath;
    bool diffuseTextureSrgb = true;
    bool emissiveTextureSrgb = true;
    glm::vec4 baseColor = glm::vec4(0.18f, 0.18f, 0.18f, 1.0f);
    glm::vec3 emissiveColor = glm::vec3(0.0f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    float opacityThreshold = 0.0f;
    bool doubleSided = false;
    Scene::MeshData meshData;
};

struct USDCameraInfo {
    std::string primPath;
    glm::vec3 position;
    glm::vec3 target;
    float verticalFov = 60.0f;
    float nearPlane = 0.01f;
    float farPlane = 1000.0f;
    float exposure = 0.0f;
    float exposureTime = 1.0f;
    float exposureIso = 100.0f;
    float exposureFStop = 1.0f;
    float exposureResponsivity = 1.0f;
    float linearExposureScale = 1.0f;
};

struct USDLightInfo {
    std::string primPath;
    Scene::LightType type = Scene::LightType::Rect;
    // The render snapshot places lights with this evaluated world transform.
    // Width/height and emission stay authored, separate from parent transforms.
    glm::mat4 worldTransform{1.0f};
    bool visible = true;
    Scene::RectLight rect;
    Scene::DomeLight dome;
};

struct USDImportResult {
    // Evaluated render snapshot, not a round-trip scene document. Source paths
    // identify USD prims; hierarchy, layers and hidden authoring data stay in USD.
    std::vector<USDMeshInfo> meshes;
    std::vector<USDCameraInfo> cameras;
    std::vector<USDLightInfo> lights;
    UpAxis upAxis = UpAxis::Y;
    // Geometry stays in authored units, multiplied by the caller's scale.
    double metersPerUnit = 0.01;
    ImportDiagnostics diagnostics;
};

struct USDArticulationImportResult {
    ArticulationDesc articulation;
    ImportDiagnostics diagnostics;
};

class USDLoader {
  public:
    USDLoader() = delete;

    static USDArticulationImportResult parseArticulation(const std::string& usdPath,
        const std::string& primPath = "", const std::string& order = "DFS");

    static USDImportResult parse(const std::string& usdPath,
                                 float scale = 1.0f);

    static std::vector<USDMeshInfo> loadMeshes(const std::string& usdPath,
                                               float scale = 1.0f);
};

} // namespace Asset
} // namespace KE

#endif

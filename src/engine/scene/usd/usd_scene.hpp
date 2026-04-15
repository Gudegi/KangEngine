///
/// USD Scene Implementation (optional)
///

#ifndef _USD_SCENE_HPP_
#define _USD_SCENE_HPP_

#ifdef KANGENGINE_USE_USD

#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>

#include "../scene_backend.hpp"

namespace KE {
namespace Scene {

class USDScene : public SceneBackend {
  private:
    pxr::UsdStageRefPtr _stage;

  public:
    USDScene();
    ~USDScene() override = default;

    // Backend interface
    BackendType getBackendType() const override { return BackendType::USD; }
    bool loadScene(const std::string& path) override;
    bool saveScene(const std::string& path) override;
    MeshData loadMesh(const std::string& primPath) override;
    std::vector<std::string> listMeshes() override;

    // ===== USD-specific API (for Python integration) =====

    /// Get the underlying USD Stage (for Python USD API)
    pxr::UsdStageRefPtr getStage() const { return _stage; }

    /// Create new in-memory stage
    void createNew();

    /// Get prim at path
    pxr::UsdPrim getPrim(const std::string& path) const;

    // USD creation helpers
    pxr::UsdPrim
    createXform(const std::string& path); // Returns UsdGeomXform prim
    pxr::UsdPrim
    createMesh(const std::string& path); // Returns UsdGeomMesh prim
    pxr::UsdPrim
    createSphere(const std::string& path,
                 double radius = 1.0); // Returns UsdGeomSphere prim
    pxr::UsdPrim createCube(const std::string& path,
                            double size = 1.0); // Returns UsdGeomCube prim

    // Utilities
    void printHierarchy() const;
};

} // namespace Scene
} // namespace KE

#endif // KANGENGINE_USE_USD
#endif // _USD_SCENE_HPP_

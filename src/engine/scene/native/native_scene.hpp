///
/// Native Scene Implementation
///

#ifndef _NATIVE_SCENE_HPP_
#define _NATIVE_SCENE_HPP_

#include "../scene_backend.hpp"
#include "prim.hpp"
#include <memory>

namespace KE {
namespace Scene {

class NativeScene : public SceneBackend {
private:
    std::unique_ptr<Prim> _root;  // "/" root prim

public:
    NativeScene();
    ~NativeScene() override = default;

    // Backend interface
    BackendType getBackendType() const override { return BackendType::Native; }
    bool loadScene(const std::string& path) override;
    bool saveScene(const std::string& path) override;
    MeshData loadMesh(const std::string& primPath) override;
    std::vector<std::string> listMeshes() override;
    Prim* definePrim(const std::string& path, PrimType type) override;

    // Scene Graph API
    Prim* getRootPrim() override { return _root.get(); }
    Prim* getPrimAtPath(const std::string& path);
    Prim* createPrim(const std::string& path, PrimType type);
};

} // namespace Scene
} // namespace KE

#endif

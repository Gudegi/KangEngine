///
/// Scene Backend Interface
///

#ifndef _SCENE_BACKEND_HPP_
#define _SCENE_BACKEND_HPP_

#include <glm/fwd.hpp>
#include <cstddef>
#include <string>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include "geometry/mesh_data.hpp"
#include "geometry/bounds.hpp"

namespace KE {
namespace Animation {
class SkeletonTree;
}
namespace Scene {

class Prim;
enum class PrimType;

// Scene Backend 타입
enum class BackendType {
    Native, // 기본 구현
    USD     // USD 구현
};

// Scene Backend 추상 인터페이스
class SceneBackend {
  public:
    virtual ~SceneBackend() = default;

    // Backend 타입 쿼리
    virtual BackendType getBackendType() const = 0;

    // Scene 로드/저장
    virtual bool loadScene(const std::string& path) = 0;
    virtual bool saveScene(const std::string& path) = 0;

    // Mesh 로드
    virtual MeshData loadMesh(const std::string& primPath) = 0;

    // Scene 쿼리
    virtual std::vector<std::string> listMeshes() = 0;

    // Prim 생성 (Scene이 소유권을 가짐, USD Stage.DefinePrim처럼)
    virtual Prim* definePrim(const std::string& path, PrimType type) {
        return nullptr;
    }
    virtual Prim* getPrimAtPath(const std::string& path) { return nullptr; }
    virtual bool removePrim(const std::string& path) { return false; }

    // Scene graph root
    virtual Prim* getRootPrim() { return nullptr; }
};

// Factory
class SceneFactory {
  public:
    static std::unique_ptr<SceneBackend> createBackend(BackendType type);
};

// Creates xform-only prims so a SkeletonTree's joint paths are visible in the
// Scene panel. This is not a skeleton visualizer or render submission path.
std::vector<Prim*> defineSkeletonTree(SceneBackend* scene,
                                      const std::string& basePath,
                                      const Animation::SkeletonTree& tree);

} // namespace Scene
} // namespace KE

#endif

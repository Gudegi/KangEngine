///
/// Scene Backend Interface
///

#ifndef _SCENE_BACKEND_HPP_
#define _SCENE_BACKEND_HPP_

#include <string>
#include <memory>
#include <vector>
#include <glm/glm.hpp>

namespace KE {
namespace Scene {

// Mesh 데이터 구조
struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;
};

// Scene Backend 타입
enum class BackendType {
    Native,  // 기본 구현
    USD      // USD 구현 (선택적)
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
};

// Factory
class SceneFactory {
public:
    static std::unique_ptr<SceneBackend> createBackend(BackendType type);
};

} // namespace Scene
} // namespace KE

#endif

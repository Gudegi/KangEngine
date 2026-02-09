///
/// Scene Backend Interface
///

#ifndef _SCENE_BACKEND_HPP_
#define _SCENE_BACKEND_HPP_

#include <glm/fwd.hpp>
#include <string>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "../backend/graphics_factory.hpp"

namespace KE {
namespace Scene {

// Forward declarations
class Prim;
enum class PrimType;

struct ShapeRenderBuffer {
    Backend::Shader* backendShader; // Backend::Shader
    std::unique_ptr<Backend::VertexArray> vertexArray;
    std::unique_ptr<Backend::Buffer> vertexBuffer;
    std::unique_ptr<Backend::Buffer> indexBuffer;
    int numIndices;

    Prim* prim; // non-owning, scene graph owns the Prim

    // Constructor helpers
    ShapeRenderBuffer() : backendShader(nullptr), numIndices(0), prim(nullptr) {}
};

// Mesh 데이터 구조
struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    MeshData() = default;

    MeshData(std::vector<glm::vec3>&& vertices,
             std::vector<glm::vec3>&& normals, std::vector<glm::vec2>&& uvs,
             std::vector<unsigned int>&& indices)
        : vertices(std::move(vertices)), normals(std::move(normals)),
          uvs(std::move(uvs)), indices(std::move(indices)) {}
};

// Scene Backend 타입
enum class BackendType {
    Native, // 기본 구현
    USD     // USD 구현
};

// Scene Backend 추상 인터페이스
class SceneBackend {
  private:
    std::vector<std::shared_ptr<ShapeRenderBuffer>> _bufferLists;

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

    // Scene graph root
    virtual Prim* getRootPrim() { return nullptr; }

    // Render 가능한 정보 쿼리
    // virtual std::vector<ShapeRenderBuffer> getRenderables() = 0;
    // virtual bool addRenderable(const std::string& path = "") = 0;
    const std::vector<std::shared_ptr<ShapeRenderBuffer>>&
    getBufferLists() const {
        return _bufferLists;
    }
    void addRenderable(std::shared_ptr<ShapeRenderBuffer> buffer) {
        _bufferLists.push_back(buffer);
    }
};

// Factory
class SceneFactory {
  public:
    static std::unique_ptr<SceneBackend> createBackend(BackendType type);
};

} // namespace Scene
} // namespace KE

#endif

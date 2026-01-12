///
/// Scene Prim (USD-like hierarchy)
///

#ifndef _SCENE_PRIM_HPP_
#define _SCENE_PRIM_HPP_

#include <string>
#include <memory>
#include <vector>
#include <map>
#include "../scene_backend.hpp"

namespace KE {
namespace Scene {

// Prim 타입
enum class PrimType {
    Root,   // "/" 루트
    Xform,  // Transform (그룹)
    Mesh,   // Mesh 데이터
    Camera, // Camera
    Light   // Light
};

// Scene Graph Node (USD Prim처럼)
class Prim {
  private:
    std::string _name; // "Cube"
    std::string _path; // "/World/Cube"
    PrimType _type;
    Prim* _parent; // 부모 (소유하지 않음)
    std::vector<std::unique_ptr<Prim>> _children; // 자식들
    std::map<std::string, Prim*> _childrenMap;    // 빠른 탐색용

    // 데이터 (타입에 따라 다름)
    std::shared_ptr<MeshData> _meshData;

  public:
    Prim(const std::string& name, PrimType type, Prim* parent = nullptr);
    ~Prim() = default;

    // 계층 구조
    Prim* addChild(const std::string& name, PrimType type);
    Prim* getChild(const std::string& name) const;
    Prim* getPrimAtPath(const std::string& path); // "/World/Cube" 경로로 탐색
    std::vector<Prim*> getChildren() const;

    // Getters
    const std::string& getName() const { return _name; }
    const std::string& getPath() const { return _path; }
    PrimType getType() const { return _type; }
    Prim* getParent() const { return _parent; }

    // Mesh 데이터 (type == Mesh일 때만)
    void setMeshData(std::shared_ptr<MeshData> data);
    std::shared_ptr<MeshData> getMeshData() const;

    static MeshData createSquareData(float scale);
    static MeshData createPlaneData(float scale);

    // 순회
    void traverse(std::function<void(Prim*)> func);
};

} // namespace Scene
} // namespace KE

#endif

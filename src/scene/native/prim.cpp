///
/// Scene Prim Implementation
///

#include "prim.hpp"
#include <sstream>

namespace KE {
namespace Scene {

Prim::Prim(const std::string& name, PrimType type, Prim* parent)
    : _name(name), _type(type), _parent(parent)
{
    // 경로 계산
    if (parent == nullptr) {
        _path = "/";  // 루트
    } else if (parent->getPath() == "/") {
        _path = "/" + name;
    } else {
        _path = parent->getPath() + "/" + name;
    }
}

Prim* Prim::addChild(const std::string& name, PrimType type) {
    auto child = std::make_unique<Prim>(name, type, this);
    Prim* childPtr = child.get();

    _childrenMap[name] = childPtr;
    _children.push_back(std::move(child));

    return childPtr;
}

Prim* Prim::getChild(const std::string& name) const {
    auto it = _childrenMap.find(name);
    if (it != _childrenMap.end()) {
        return it->second;
    }
    return nullptr;
}

Prim* Prim::getPrimAtPath(const std::string& path) {
    // "/" → 루트
    if (path == "/" || path.empty()) {
        return this;
    }

    // "/World/Cube" 파싱
    std::string pathCopy = path;
    if (pathCopy[0] == '/') {
        pathCopy = pathCopy.substr(1);  // "/" 제거
    }

    // "World/Cube" → ["World", "Cube"]
    std::vector<std::string> parts;
    std::stringstream ss(pathCopy);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }

    // 순차 탐색
    Prim* current = this;
    for (const auto& name : parts) {
        current = current->getChild(name);
        if (!current) {
            return nullptr;  // 경로 없음
        }
    }

    return current;
}

std::vector<Prim*> Prim::getChildren() const {
    std::vector<Prim*> result;
    result.reserve(_children.size());
    for (const auto& child : _children) {
        result.push_back(child.get());
    }
    return result;
}

void Prim::setMeshData(std::shared_ptr<MeshData> data) {
    _meshData = data;
}

std::shared_ptr<MeshData> Prim::getMeshData() const {
    return _meshData;
}

MeshData Prim::createSquareData(float scale)
{
    // Scale means the length of one side.
    float half = scale / 2;
    //    v3----- v7  
    //   /|      /|   
    //  v2------v6| 
    //  | |     | |  
    //  | v0----|-v4  
    //  |/      |/
    //  v1------v5 
    //      
    std::vector<glm::vec3> positions = {
        // v0, v1, v2, v3
        glm::vec3(-half, -half, -half), glm::vec3(-half, -half, half), glm::vec3(-half, half, half), glm::vec3(-half, half, -half),
        // v4, v5, v6, v7
        glm::vec3(half, -half, -half), glm::vec3(half, -half, half), glm::vec3(half, half, half), glm::vec3(half, half, -half),
        // v0, v1, v5, v4
        glm::vec3(-half, -half, -half), glm::vec3(-half, -half, half), glm::vec3(half, -half, half), glm::vec3(half, -half, -half),
        // v3, v2, v6, v7
        glm::vec3(-half, half, -half), glm::vec3(-half, half, half), glm::vec3(half, half, half), glm::vec3(half, half, -half),
        // v0, v3, v7, v4
        glm::vec3(-half, -half, -half), glm::vec3(-half, half, -half), glm::vec3(half, half, -half), glm::vec3(half, -half, -half),
        // v1, v2, v6, v5
        glm::vec3(-half, -half, half), glm::vec3(-half, half, half), glm::vec3(half, half, half), glm::vec3(half, -half, half),
    }; // positions.size() == 24

    std::vector<glm::vec3> normals = {
        glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0),
        glm::vec3(1, 0, 0), glm::vec3(1, 0, 0), glm::vec3(1, 0, 0), glm::vec3(1, 0, 0),
        glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
        glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0),
        glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, -1),
        glm::vec3(0, 0, 1), glm::vec3(0, 0, 1), glm::vec3(0, 0, 1), glm::vec3(0, 0, 1),
    };

    std::vector<glm::vec2> uvs = {
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
    };

    std::vector<unsigned int> indices = {
        0, 1, 2, 0, 2, 3, // left 
        4, 6, 5, 4, 7, 6, // right
        8, 10, 9, 8, 11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 22, 21, 20, 23, 22, // front
    };

    MeshData meshData;
    meshData.vertices = positions;
    meshData.normals = normals;
    meshData.uvs = uvs;
    meshData.indices = indices;

    return meshData;
}

void Prim::traverse(std::function<void(Prim*)> func) {
    func(this);
    for (auto& child : _children) {
        child->traverse(func);
    }
}

} // namespace Scene
} // namespace KE

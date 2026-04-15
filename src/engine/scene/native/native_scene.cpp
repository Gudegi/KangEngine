///
/// Native Scene Implementation
///

#include "native_scene.hpp"
#include <iostream>

namespace KE {
namespace Scene {

NativeScene::NativeScene() {
    // 루트 prim 생성
    _root = std::make_unique<Prim>("", PrimType::Root, nullptr);
}

bool NativeScene::loadScene(const std::string& path) {
    // TODO: JSON/XML에서 scene 로드하여 prim 구조 생성
    std::cout << "NativeScene::loadScene - not implemented yet" << std::endl;
    return false;
}

bool NativeScene::saveScene(const std::string& path) {
    // TODO: Prim 구조를 JSON/XML로 저장
    std::cout << "NativeScene::saveScene - not implemented yet" << std::endl;
    return false;
}

MeshData NativeScene::loadMesh(const std::string& primPath) {
    Prim* prim = _root->getPrimAtPath(primPath);
    if (prim && prim->getType() == PrimType::Mesh) {
        auto data = prim->getMeshData();
        if (data) {
            return *data;
        }
    }
    return MeshData();
}

std::vector<std::string> NativeScene::listMeshes() {
    std::vector<std::string> meshPaths;

    // Scene graph 순회하여 모든 mesh 찾기
    _root->traverse([&meshPaths](Prim* prim) {
        if (prim->getType() == PrimType::Mesh) {
            meshPaths.push_back(prim->getPath());
        }
    });

    return meshPaths;
}

Prim* NativeScene::definePrim(const std::string& path, PrimType type) {
    return createPrim(path, type);
}

Prim* NativeScene::getPrimAtPath(const std::string& path) {
    return _root->getPrimAtPath(path);
}

Prim* NativeScene::createPrim(const std::string& path, PrimType type) {
    // "/World/Cube" → [World, Cube]
    std::string pathCopy = path;
    if (pathCopy[0] == '/') {
        pathCopy = pathCopy.substr(1);
    }

    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = pathCopy.find('/');
    while (end != std::string::npos) {
        parts.push_back(pathCopy.substr(start, end - start));
        start = end + 1;
        end = pathCopy.find('/', start);
    }
    parts.push_back(pathCopy.substr(start));

    // 부모까지 생성
    Prim* current = _root.get();
    for (size_t i = 0; i < parts.size() - 1; ++i) {
        Prim* child = current->getChild(parts[i]);
        if (!child) {
            child = current->addChild(parts[i], PrimType::Xform);
        }
        current = child;
    }

    // 마지막 prim 생성
    return current->addChild(parts.back(), type);
}

} // namespace Scene
} // namespace KE

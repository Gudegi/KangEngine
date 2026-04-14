///
/// Scene Graph (USD-like) Example
///

#include "kangEngine.hpp"
#include "scene/native/native_scene.hpp"
#include "scene/native/prim.hpp"
#include <iostream>

int main() {
    using namespace KE::Scene;

    // Native Scene 생성
    NativeScene scene;

    // USD처럼 계층 구조 생성
    // /
    // └── World
    //     ├── Cube
    //     └── Sphere

    Prim* worldPrim = scene.createPrim("/World", PrimType::Xform);
    std::cout << "Created: " << worldPrim->getPath() << std::endl;

    Prim* cubePrim = scene.createPrim("/World/Cube", PrimType::Mesh);
    std::cout << "Created: " << cubePrim->getPath() << std::endl;

    Prim* spherePrim = scene.createPrim("/World/Sphere", PrimType::Mesh);
    std::cout << "Created: " << spherePrim->getPath() << std::endl;

    // Cube에 mesh 데이터 추가
    auto cubeData = std::make_shared<MeshData>(Prim::createSquareData(1.0f));
    cubePrim->setMeshData(cubeData);

    // 경로로 prim 찾기
    Prim* foundCube = scene.getPrimAtPath("/World/Cube");
    if (foundCube) {
        std::cout << "Found prim: " << foundCube->getPath() << std::endl;
        std::cout << "Type: " << (foundCube->getType() == PrimType::Mesh ? "Mesh" : "Other") << std::endl;

        auto meshData = foundCube->getMeshData();
        if (meshData) {
            std::cout << "Vertices: " << meshData->vertices.size() << std::endl;
        }
    }

    // 모든 mesh 나열
    std::cout << "\nAll meshes in scene:" << std::endl;
    auto meshes = scene.listMeshes();
    for (const auto& meshPath : meshes) {
        std::cout << "  - " << meshPath << std::endl;
    }

    // Scene graph 순회
    std::cout << "\nScene hierarchy:" << std::endl;
    scene.getRootPrim()->traverse([](Prim* prim) {
        std::cout << "  " << prim->getPath() << std::endl;
    });

    return 0;
}

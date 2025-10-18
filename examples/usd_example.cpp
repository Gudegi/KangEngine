///
/// USD Scene Loading Example
///

#include "kangEngine.hpp"
#include "scene/scene_backend.hpp"
#include <iostream>

int main() {
#ifdef KANGENGINE_USE_USD
    std::cout << "USD support enabled" << std::endl;

    // USD Scene 백엔드 생성
    auto sceneBackend = KE::Scene::SceneFactory::createBackend(KE::Scene::BackendType::USD);

    // USD 파일 로드
    if (sceneBackend->loadScene("/Users/asaid/Downloads/test.usd")) {
        // 모든 mesh 리스트
        auto meshes = sceneBackend->listMeshes();
        std::cout << "Found " << meshes.size() << " meshes:" << std::endl;
        for (const auto& meshPath : meshes) {
            std::cout << "  - " << meshPath << std::endl;
        }

        // 특정 mesh 로드
        if (!meshes.empty()) {
            auto meshData = sceneBackend->loadMesh(meshes[0]);
            std::cout << "Loaded mesh with " << meshData.vertices.size() << " vertices" << std::endl;

            // TODO: App에 추가
            // app.addMesh(meshData.vertices, meshData.indices, ...);
        }
    }
#else
    std::cout << "USD support not enabled. Rebuild with -DUSE_USD=ON" << std::endl;

    // Native Scene 사용
    auto sceneBackend = KE::Scene::SceneFactory::createBackend(KE::Scene::BackendType::Native);
    std::cout << "Using native scene backend" << std::endl;
#endif

    return 0;
}

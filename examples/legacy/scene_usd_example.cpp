///
/// USD Scene Example - Python Integration Ready
/// This demonstrates how to use USDScene with direct Stage access
/// for Python USD API compatibility
///

#include "kangEngine.hpp"
#include "engine/scene/scene_backend.hpp"

#ifdef KANGENGINE_USE_USD
#include "engine/scene/usd/usd_scene.hpp"
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#endif

#include <iostream>

int main() {
#ifdef KANGENGINE_USE_USD
    std::cout << "=== USD Scene Example ===" << std::endl;

    // Create USD Scene backend
    auto sceneBackend = KE::Scene::SceneFactory::createBackend(KE::Scene::BackendType::USD);

    // Cast to USDScene to access USD-specific API
    KE::Scene::USDScene* usdScene = dynamic_cast<KE::Scene::USDScene*>(sceneBackend.get());

    if (!usdScene) {
        std::cerr << "Failed to create USD scene" << std::endl;
        return 1;
    }

    std::cout << "\n1. Creating scene with USD API..." << std::endl;

    // Method 1: Use convenience methods
    usdScene->createXform("/World");
    usdScene->createSphere("/World/Sphere1", 2.0);
    usdScene->createCube("/World/Cube1", 1.5);

    // Method 2: Direct USD Stage access 
    std::cout << "\n2. Direct USD Stage access..." << std::endl;
    pxr::UsdStageRefPtr stage = usdScene->getStage();

    if (stage) {
        pxr::UsdGeomXform group = pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/World/Group"));

        pxr::UsdGeomSphere sphere2 = pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/World/Group/Sphere2"));
        sphere2.GetRadiusAttr().Set(1.0);

        // Add transform
        group.AddTranslateOp().Set(pxr::GfVec3d(5.0, 0.0, 0.0));

        std::cout << "Created prims using direct USD API" << std::endl;
    }

    // Print hierarchy
    std::cout << "\n3. Scene Hierarchy:" << std::endl;
    usdScene->printHierarchy();

    // List all meshes (using backend interface)
    std::cout << "\n4. Meshes in scene:" << std::endl;
    auto meshes = usdScene->listMeshes();
    for (const auto& meshPath : meshes) {
        std::cout << "  - " << meshPath << std::endl;
    }

    // Save to file
    std::cout << "\n5. Saving to USD file..." << std::endl;
    std::string outputPath = "output_scene.usda";
    if (usdScene->saveScene(outputPath)) {
        std::cout << "Saved to: " << outputPath << std::endl;
        std::cout << "You can now open this in usdview or edit with Python USD API!" << std::endl;
    }

    // Demo: Load existing USD file
    std::cout << "\n6. Loading USD file (if exists)..." << std::endl;
    std::string inputPath = "/Users/asaid/Downloads/test.usd";

    auto loadedScene = KE::Scene::SceneFactory::createBackend(KE::Scene::BackendType::USD);
    if (loadedScene->loadScene(inputPath)) {
        KE::Scene::USDScene* loadedUSD = dynamic_cast<KE::Scene::USDScene*>(loadedScene.get());
        if (loadedUSD) {
            std::cout << "Loaded scene hierarchy:" << std::endl;
            loadedUSD->printHierarchy();

            // Get Stage for further processing
            pxr::UsdStageRefPtr loadedStage = loadedUSD->getStage();
            std::cout << "Stage accessible for Python USD API!" << std::endl;
        }
    } else {
        std::cout << "File not found (this is OK for demo)" << std::endl;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "This example shows:" << std::endl;
    std::cout << "1. Creating scenes with convenience methods" << std::endl;
    std::cout << "2. Direct USD Stage access via getStage()" << std::endl;
    std::cout << "3. Full USD API compatibility" << std::endl;
    std::cout << "4. Python integration ready!" << std::endl;
    std::cout << "\nIn Python, you would do:" << std::endl;
    std::cout << "  scene = ke.Scene.createUSD()" << std::endl;
    std::cout << "  stage = scene.getStage()  # Returns pxr.UsdStage" << std::endl;
    std::cout << "  xform = UsdGeom.Xform.Define(stage, '/MyXform')" << std::endl;

#else
    std::cout << "USD support not enabled." << std::endl;
    std::cout << "Rebuild with: cmake --preset=vcpkg -DUSE_USD=ON" << std::endl;
    std::cout << "Then: cmake --build build" << std::endl;
#endif

    return 0;
}

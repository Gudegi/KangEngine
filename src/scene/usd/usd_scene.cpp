///
/// USD Scene Implementation
///

#ifdef KANGENGINE_USE_USD

#include "usd_scene.hpp"
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/base/vt/array.h>
#include <iostream>

namespace KE {
namespace Scene {

USDScene::USDScene() {
    // Create in-memory stage by default
    createNew();
}

bool USDScene::loadScene(const std::string& path) {
    _stage = pxr::UsdStage::Open(path);
    if (!_stage) {
        std::cerr << "Failed to open USD stage: " << path << std::endl;
        return false;
    }
    std::cout << "Loaded USD stage: " << path << std::endl;
    return true;
}

bool USDScene::saveScene(const std::string& path) {
    if (!_stage) {
        std::cerr << "No stage to save" << std::endl;
        return false;
    }
    return _stage->Export(path);
}

MeshData USDScene::loadMesh(const std::string& primPath) {
    MeshData meshData;

    if (!_stage) {
        std::cerr << "No stage loaded" << std::endl;
        return meshData;
    }

    pxr::UsdPrim prim = _stage->GetPrimAtPath(pxr::SdfPath(primPath));
    if (!prim) {
        std::cerr << "Prim not found: " << primPath << std::endl;
        return meshData;
    }

    pxr::UsdGeomMesh mesh(prim);
    if (!mesh) {
        std::cerr << "Prim is not a mesh: " << primPath << std::endl;
        return meshData;
    }

    // Points (vertices)
    pxr::VtArray<pxr::GfVec3f> points;
    mesh.GetPointsAttr().Get(&points);
    meshData.vertices.reserve(points.size());
    for (const auto& p : points) {
        meshData.vertices.push_back(glm::vec3(p[0], p[1], p[2]));
    }

    // Normals
    pxr::VtArray<pxr::GfVec3f> normals;
    mesh.GetNormalsAttr().Get(&normals);
    meshData.normals.reserve(normals.size());
    for (const auto& n : normals) {
        meshData.normals.push_back(glm::vec3(n[0], n[1], n[2]));
    }

    // Indices
    pxr::VtArray<int> faceVertexIndices;
    mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);
    meshData.indices.reserve(faceVertexIndices.size());
    for (const auto& idx : faceVertexIndices) {
        meshData.indices.push_back(static_cast<unsigned int>(idx));
    }

    std::cout << "Loaded mesh: " << primPath
              << " (vertices: " << meshData.vertices.size()
              << ", indices: " << meshData.indices.size() << ")" << std::endl;

    return meshData;
}

std::vector<std::string> USDScene::listMeshes() {
    std::vector<std::string> meshPaths;

    if (!_stage) {
        return meshPaths;
    }

    // USD stage 순회하여 모든 mesh 찾기
    pxr::UsdPrimRange range = _stage->Traverse();
    for (const auto& prim : range) {
        if (prim.IsA<pxr::UsdGeomMesh>()) {
            meshPaths.push_back(prim.GetPath().GetString());
        }
    }

    return meshPaths;
}

// ===== USD-specific API =====

void USDScene::createNew() {
    _stage = pxr::UsdStage::CreateInMemory();
    if (_stage) {
        // Create default root xform
        pxr::UsdGeomXform root = pxr::UsdGeomXform::Define(_stage, pxr::SdfPath("/Root"));
        _stage->SetDefaultPrim(root.GetPrim());
        std::cout << "Created new USD stage with /Root" << std::endl;
    }
}

pxr::UsdPrim USDScene::getPrim(const std::string& path) const {
    if (!_stage) {
        return pxr::UsdPrim();
    }
    return _stage->GetPrimAtPath(pxr::SdfPath(path));
}

pxr::UsdPrim USDScene::createXform(const std::string& path) {
    if (!_stage) {
        std::cerr << "No stage available" << std::endl;
        return pxr::UsdPrim();
    }
    pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(_stage, pxr::SdfPath(path));
    return xform.GetPrim();
}

pxr::UsdPrim USDScene::createMesh(const std::string& path) {
    if (!_stage) {
        std::cerr << "No stage available" << std::endl;
        return pxr::UsdPrim();
    }
    pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh::Define(_stage, pxr::SdfPath(path));
    return mesh.GetPrim();
}

pxr::UsdPrim USDScene::createSphere(const std::string& path, double radius) {
    if (!_stage) {
        std::cerr << "No stage available" << std::endl;
        return pxr::UsdPrim();
    }
    pxr::UsdGeomSphere sphere = pxr::UsdGeomSphere::Define(_stage, pxr::SdfPath(path));
    sphere.GetRadiusAttr().Set(radius);
    return sphere.GetPrim();
}

pxr::UsdPrim USDScene::createCube(const std::string& path, double size) {
    if (!_stage) {
        std::cerr << "No stage available" << std::endl;
        return pxr::UsdPrim();
    }
    pxr::UsdGeomCube cube = pxr::UsdGeomCube::Define(_stage, pxr::SdfPath(path));
    cube.GetSizeAttr().Set(size);
    return cube.GetPrim();
}

void USDScene::printHierarchy() const {
    if (!_stage) {
        std::cout << "No stage loaded" << std::endl;
        return;
    }

    std::cout << "=== USD Scene Hierarchy ===" << std::endl;

    std::function<void(const pxr::UsdPrim&, int)> printPrim =
        [&](const pxr::UsdPrim& prim, int depth) {

        // Indentation
        for (int i = 0; i < depth; ++i) {
            std::cout << "  ";
        }

        // Prim info
        std::cout << prim.GetName() << " [" << prim.GetTypeName() << "]";

        if (prim == _stage->GetDefaultPrim()) {
            std::cout << " (default)";
        }

        std::cout << std::endl;

        // Children
        for (const auto& child : prim.GetChildren()) {
            printPrim(child, depth + 1);
        }
    };

    printPrim(_stage->GetPseudoRoot(), 0);
    std::cout << "===========================" << std::endl;
}

} // namespace Scene
} // namespace KE

#endif // KANGENGINE_USE_USD

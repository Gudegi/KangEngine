///
/// USD Scene Implementation
///

#ifdef KANGENGINE_USE_USD

#include "usd_scene.hpp"
#include "geometry/mesh_utils.hpp"
#include <iostream>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCache.h>

namespace KE {
namespace Scene {
namespace {

glm::vec3 transformPoint(const pxr::GfMatrix4d& matrix,
                         const pxr::GfVec3f& point) {
    const pxr::GfVec3d world =
        matrix.Transform(pxr::GfVec3d(point[0], point[1], point[2]));
    return glm::vec3(world[0], world[1], world[2]);
}

glm::vec3 transformNormal(const pxr::GfMatrix4d& matrix,
                          const pxr::GfVec3f& normal) {
    const pxr::GfVec3d world =
        matrix.TransformDir(pxr::GfVec3d(normal[0], normal[1], normal[2]));
    const glm::vec3 out(world[0], world[1], world[2]);
    const float len = glm::length(out);
    return len > 1e-8f ? out / len : glm::vec3(0.0f, 1.0f, 0.0f);
}

bool readIndexedValueIndex(const pxr::VtArray<int>& indices, size_t srcIndex,
                           size_t valueCount, size_t& outIndex) {
    if (!indices.empty()) {
        if (srcIndex >= indices.size() || indices[srcIndex] < 0)
            return false;
        outIndex = static_cast<size_t>(indices[srcIndex]);
    } else {
        outIndex = srcIndex;
    }
    return outIndex < valueCount;
}

} // namespace

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

    pxr::VtArray<int> faceVertexCounts;
    mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);

    pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());
    const pxr::GfMatrix4d localToWorld =
        xformCache.GetLocalToWorldTransform(prim);
    const pxr::GfMatrix4d normalToWorld =
        localToWorld.GetInverse().GetTranspose();

    pxr::VtArray<pxr::GfVec3f> points;
    mesh.GetPointsAttr().Get(&points);

    pxr::VtArray<pxr::GfVec3f> normals;
    mesh.GetNormalsAttr().Get(&normals);
    const pxr::TfToken normalInterpolation = mesh.GetNormalsInterpolation();

    pxr::VtArray<pxr::GfVec2f> uvs;
    pxr::VtArray<int> uvIndices;
    pxr::TfToken uvInterpolation;
    pxr::UsdGeomPrimvar st =
        pxr::UsdGeomPrimvarsAPI(prim).GetPrimvar(pxr::TfToken("st"));
    if (st) {
        st.Get(&uvs);
        st.GetIndices(&uvIndices);
        uvInterpolation = st.GetInterpolation();
    }

    pxr::VtArray<int> faceVertexIndices;
    mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);

    size_t triangulatedIndexCount = 0;
    for (const int count : faceVertexCounts) {
        if (count < 3) {
            std::cerr << "USDScene::loadMesh skipping degenerate face in "
                      << primPath << " with " << count << " vertices"
                      << std::endl;
            continue;
        }
        triangulatedIndexCount += static_cast<size_t>(count - 2) * 3;
    }
    meshData.vertices.reserve(triangulatedIndexCount);
    meshData.indices.reserve(triangulatedIndexCount);
    if (!normals.empty())
        meshData.normals.reserve(triangulatedIndexCount);
    if (!uvs.empty())
        meshData.uvs.reserve(triangulatedIndexCount);

    const bool normalsFaceVarying =
        normalInterpolation == pxr::TfToken("faceVarying") &&
        normals.size() == faceVertexIndices.size();
    const bool normalsVertex =
        !normals.empty() && normals.size() == points.size();
    const bool useNormals = normalsFaceVarying || normalsVertex;

    const bool uvFaceVarying =
        uvInterpolation == pxr::TfToken("faceVarying") ||
        uvs.size() == faceVertexIndices.size() ||
        (!uvIndices.empty() && uvIndices.size() == faceVertexIndices.size());
    const bool uvVertex = !uvs.empty() && uvs.size() == points.size();
    const bool useUvs = !uvs.empty();

    auto emitCorner = [&](size_t cornerOffset) -> bool {
        if (cornerOffset >= faceVertexIndices.size() ||
            faceVertexIndices[cornerOffset] < 0)
            return false;
        const size_t pointIndex =
            static_cast<size_t>(faceVertexIndices[cornerOffset]);
        if (pointIndex >= points.size())
            return false;

        meshData.vertices.push_back(
            transformPoint(localToWorld, points[pointIndex]));

        if (useNormals) {
            const size_t normalIndex =
                normalsFaceVarying ? cornerOffset : pointIndex;
            meshData.normals.push_back(
                transformNormal(normalToWorld, normals[normalIndex]));
        }

        if (useUvs) {
            glm::vec2 uv(0.0f);
            size_t uvIndex = 0;
            if (uvFaceVarying && readIndexedValueIndex(uvIndices, cornerOffset,
                                                       uvs.size(), uvIndex)) {
                uv = glm::vec2(uvs[uvIndex][0], uvs[uvIndex][1]);
            } else if (uvVertex && pointIndex < uvs.size()) {
                uv = glm::vec2(uvs[pointIndex][0], uvs[pointIndex][1]);
            } else if (uvs.size() == 1) {
                uv = glm::vec2(uvs[0][0], uvs[0][1]);
            }
            meshData.uvs.push_back(uv);
        }
        return true;
    };

    size_t offset = 0;
    for (const int count : faceVertexCounts) {
        if (count < 3) {
            offset += static_cast<size_t>(std::max(0, count));
            continue;
        }
        if (offset + static_cast<size_t>(count) > faceVertexIndices.size()) {
            std::cerr << "USDScene::loadMesh face index data ended early for "
                      << primPath << std::endl;
            meshData.indices.clear();
            return meshData;
        }

        for (int i = 1; i + 1 < count; ++i) {
            const unsigned int base =
                static_cast<unsigned int>(meshData.vertices.size());
            if (!emitCorner(offset) ||
                !emitCorner(offset + static_cast<size_t>(i)) ||
                !emitCorner(offset + static_cast<size_t>(i + 1))) {
                std::cerr << "USDScene::loadMesh found invalid face index for "
                          << primPath << std::endl;
                meshData = MeshData();
                return meshData;
            }
            meshData.indices.push_back(base);
            meshData.indices.push_back(base + 1);
            meshData.indices.push_back(base + 2);
        }
        offset += static_cast<size_t>(count);
    }

    meshData.fillMissingAttributes();
    Geometry::computeTangents(meshData);

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
        pxr::UsdGeomXform root =
            pxr::UsdGeomXform::Define(_stage, pxr::SdfPath("/Root"));
        _stage->SetDefaultPrim(root.GetPrim());
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
    pxr::UsdGeomXform xform =
        pxr::UsdGeomXform::Define(_stage, pxr::SdfPath(path));
    return xform.GetPrim();
}

pxr::UsdPrim USDScene::createMesh(const std::string& path) {
    if (!_stage) {
        std::cerr << "No stage available" << std::endl;
        return pxr::UsdPrim();
    }
    pxr::UsdGeomMesh mesh =
        pxr::UsdGeomMesh::Define(_stage, pxr::SdfPath(path));
    return mesh.GetPrim();
}

pxr::UsdPrim USDScene::createSphere(const std::string& path, double radius) {
    if (!_stage) {
        std::cerr << "No stage available" << std::endl;
        return pxr::UsdPrim();
    }
    pxr::UsdGeomSphere sphere =
        pxr::UsdGeomSphere::Define(_stage, pxr::SdfPath(path));
    sphere.GetRadiusAttr().Set(radius);
    return sphere.GetPrim();
}

pxr::UsdPrim USDScene::createCube(const std::string& path, double size) {
    if (!_stage) {
        std::cerr << "No stage available" << std::endl;
        return pxr::UsdPrim();
    }
    pxr::UsdGeomCube cube =
        pxr::UsdGeomCube::Define(_stage, pxr::SdfPath(path));
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

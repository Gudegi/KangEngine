#include "usd_loader.hpp"

#include "asset/mesh_loader.hpp"
#include "geometry/mesh_utils.hpp"

#include <filesystem>
#include <stdexcept>
#include <unordered_set>

#ifdef KANGENGINE_USE_USD
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#endif

namespace KE {
namespace Asset {
namespace {

std::string primNameFromPath(const std::string& primPath) {
    const size_t pos = primPath.find_last_of('/');
    if (pos == std::string::npos)
        return primPath;
    if (pos + 1 >= primPath.size())
        return {};
    return primPath.substr(pos + 1);
}

void applyScale(Scene::MeshData& mesh, float scale) {
    if (scale == 1.0f)
        return;
    for (glm::vec3& vertex : mesh.vertices)
        vertex *= scale;
}

#ifdef KANGENGINE_USE_USD
std::string assetPathString(const pxr::SdfAssetPath& path) {
    if (!path.GetResolvedPath().empty())
        return path.GetResolvedPath();
    return path.GetAssetPath();
}

std::string resolveUsdTexturePath(const std::string& usdPath,
                                  const std::string& assetPath) {
    if (assetPath.empty())
        return {};
    std::filesystem::path path(assetPath);
    if (path.is_absolute())
        return path.string();

    std::string normalized = assetPath;
    for (char& ch : normalized) {
        if (ch == '\\')
            ch = '/';
    }
    return (std::filesystem::path(usdPath).parent_path() / normalized).string();
}

std::string findTextureInShaderNetwork(const pxr::UsdPrim& prim,
                                       const std::string& usdPath) {
    if (!prim)
        return {};

    pxr::UsdShadeShader shader(prim);
    if (shader) {
        pxr::UsdShadeInput fileInput = shader.GetInput(pxr::TfToken("file"));
        if (fileInput) {
            pxr::SdfAssetPath asset;
            if (fileInput.Get(&asset)) {
                return resolveUsdTexturePath(usdPath, assetPathString(asset));
            }
        }
    }

    for (const pxr::UsdPrim& child : prim.GetChildren()) {
        std::string path = findTextureInShaderNetwork(child, usdPath);
        if (!path.empty())
            return path;
    }
    return {};
}

std::string connectedTexturePath(const pxr::UsdShadeMaterial& material,
                                 const std::string& usdPath,
                                 const pxr::TfToken& inputName) {
    if (!material)
        return {};

    pxr::UsdShadeShader shader = material.ComputeSurfaceSource();
    if (!shader)
        return {};

    pxr::UsdShadeInput input = shader.GetInput(inputName);
    if (!input)
        return {};

    const pxr::UsdShadeInput::SourceInfoVector sources =
        input.GetConnectedSources();
    if (sources.empty())
        return {};

    return findTextureInShaderNetwork(sources.front().source.GetPrim(),
                                      usdPath);
}

std::string diffuseTexturePath(const pxr::UsdShadeMaterial& material,
                               const std::string& usdPath) {
    return connectedTexturePath(material, usdPath,
                                pxr::TfToken("diffuseColor"));
}

std::string normalTexturePath(const pxr::UsdShadeMaterial& material,
                              const std::string& usdPath) {
    return connectedTexturePath(material, usdPath, pxr::TfToken("normal"));
}

pxr::UsdShadeMaterial boundMaterial(const pxr::UsdPrim& prim) {
    pxr::UsdShadeMaterialBindingAPI binding(prim);
    return binding.ComputeBoundMaterial();
}

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

Scene::MeshData loadMeshData(const pxr::UsdPrim& prim,
                             const std::unordered_set<size_t>* selectedFaces) {
    Scene::MeshData meshData;
    pxr::UsdGeomMesh mesh(prim);
    if (!mesh)
        return meshData;

    pxr::VtArray<int> faceVertexCounts;
    pxr::VtArray<int> faceVertexIndices;
    pxr::VtArray<pxr::GfVec3f> points;
    mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);
    mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);
    mesh.GetPointsAttr().Get(&points);
    if (faceVertexCounts.empty() || faceVertexIndices.empty() || points.empty())
        return meshData;

    pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());
    const pxr::GfMatrix4d localToWorld =
        xformCache.GetLocalToWorldTransform(prim);
    const pxr::GfMatrix4d normalToWorld =
        localToWorld.GetInverse().GetTranspose();

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

    std::vector<size_t> faceOffsets(faceVertexCounts.size(), 0);
    size_t offset = 0;
    size_t triangulatedIndexCount = 0;
    for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
        faceOffsets[face] = offset;
        const int count = faceVertexCounts[face];
        if (count > 0)
            offset += static_cast<size_t>(count);
        if (selectedFaces && selectedFaces->count(face) == 0)
            continue;
        if (count >= 3)
            triangulatedIndexCount += static_cast<size_t>(count - 2) * 3;
    }
    if (offset > faceVertexIndices.size())
        return meshData;

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

    for (size_t face = 0; face < faceVertexCounts.size(); ++face) {
        if (selectedFaces && selectedFaces->count(face) == 0)
            continue;
        const int count = faceVertexCounts[face];
        if (count < 3)
            continue;

        const size_t faceOffset = faceOffsets[face];
        for (int i = 1; i + 1 < count; ++i) {
            const unsigned int base =
                static_cast<unsigned int>(meshData.vertices.size());
            if (!emitCorner(faceOffset) ||
                !emitCorner(faceOffset + static_cast<size_t>(i)) ||
                !emitCorner(faceOffset + static_cast<size_t>(i + 1))) {
                return Scene::MeshData();
            }
            meshData.indices.push_back(base);
            meshData.indices.push_back(base + 1);
            meshData.indices.push_back(base + 2);
        }
    }

    meshData.fillMissingAttributes();
    Scene::MeshData deduped = deduplicateMeshData(meshData);
    Geometry::computeTangents(deduped);
    return deduped;
}

std::unordered_set<size_t> faceSetFromSubset(const pxr::UsdPrim& subsetPrim) {
    std::unordered_set<size_t> faces;
    pxr::VtArray<int> indices;
    subsetPrim.GetAttribute(pxr::TfToken("indices")).Get(&indices);
    faces.reserve(indices.size());
    for (const int index : indices) {
        if (index >= 0)
            faces.insert(static_cast<size_t>(index));
    }
    return faces;
}

size_t faceCount(const pxr::UsdPrim& prim) {
    pxr::VtArray<int> faceVertexCounts;
    pxr::UsdGeomMesh(prim).GetFaceVertexCountsAttr().Get(&faceVertexCounts);
    return faceVertexCounts.size();
}

void assignMaterial(USDMeshInfo& info, const pxr::UsdShadeMaterial& material,
                    const std::string& usdPath, ImportDiagnostics& diagnostics,
                    std::unordered_set<std::string>& missingTextureWarnings) {
    if (!material)
        return;
    info.materialPath = material.GetPath().GetString();
    info.diffuseTexturePath = diffuseTexturePath(material, usdPath);
    info.normalTexturePath = normalTexturePath(material, usdPath);
    if (info.diffuseTexturePath.empty() &&
        missingTextureWarnings.insert(info.materialPath).second) {
        diagnostics.warnings.push_back(
            "USD diffuse texture was not found for " + info.materialPath);
    }
}
#endif

} // namespace

USDImportResult USDLoader::parse(const std::string& usdPath, float scale) {
    USDImportResult result;

#ifndef KANGENGINE_USE_USD
    (void)scale;
    throw std::runtime_error(
        "USD support not compiled. Rebuild with -DUSE_USD=ON");
#else
    if (!std::filesystem::exists(usdPath)) {
        throw std::runtime_error("USD file does not exist: " + usdPath);
    }

    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(usdPath);
    if (!stage) {
        throw std::runtime_error("Failed to load USD scene: " + usdPath);
    }

    std::unordered_set<std::string> missingTextureWarnings;
    for (const pxr::UsdPrim& prim : stage->Traverse()) {
        if (!prim.IsA<pxr::UsdGeomMesh>())
            continue;

        const std::string primPath = prim.GetPath().GetString();
        const size_t numFaces = faceCount(prim);
        std::vector<bool> coveredFaces(numFaces, false);
        size_t emittedSubsetCount = 0;

        for (const pxr::UsdPrim& child : prim.GetChildren()) {
            if (!child.IsA<pxr::UsdGeomSubset>())
                continue;

            const pxr::UsdShadeMaterial material = boundMaterial(child);
            if (!material)
                continue;

            const std::unordered_set<size_t> faces = faceSetFromSubset(child);
            if (faces.empty())
                continue;

            Scene::MeshData meshData = loadMeshData(prim, &faces);
            if (meshData.vertices.empty() || meshData.indices.empty()) {
                result.diagnostics.warnings.push_back(
                    "Skipped unsupported or empty USD material subset: " +
                    child.GetPath().GetString());
                continue;
            }

            for (const size_t face : faces) {
                if (face < coveredFaces.size())
                    coveredFaces[face] = true;
            }

            applyScale(meshData, scale);
            USDMeshInfo info;
            info.primPath = child.GetPath().GetString();
            info.name = primNameFromPath(primPath) + "_" +
                        primNameFromPath(info.primPath);
            info.meshData = std::move(meshData);
            assignMaterial(info, material, usdPath, result.diagnostics,
                           missingTextureWarnings);
            result.meshes.push_back(std::move(info));
            ++emittedSubsetCount;
        }

        if (emittedSubsetCount > 0) {
            std::unordered_set<size_t> remainingFaces;
            for (size_t face = 0; face < coveredFaces.size(); ++face) {
                if (!coveredFaces[face])
                    remainingFaces.insert(face);
            }
            if (remainingFaces.empty())
                continue;

            Scene::MeshData meshData = loadMeshData(prim, &remainingFaces);
            if (meshData.vertices.empty() || meshData.indices.empty()) {
                result.diagnostics.warnings.push_back(
                    "Skipped unsupported or empty USD unbound subset: " +
                    primPath);
                continue;
            }

            applyScale(meshData, scale);
            USDMeshInfo info;
            info.primPath = primPath;
            info.name = primNameFromPath(primPath) + "_unbound";
            info.meshData = std::move(meshData);
            assignMaterial(info, boundMaterial(prim), usdPath,
                           result.diagnostics, missingTextureWarnings);
            result.meshes.push_back(std::move(info));
            continue;
        }

        Scene::MeshData meshData = loadMeshData(prim, nullptr);
        if (meshData.vertices.empty() || meshData.indices.empty()) {
            result.diagnostics.warnings.push_back(
                "Skipped unsupported or empty USD mesh: " + primPath);
            continue;
        }

        applyScale(meshData, scale);
        USDMeshInfo info;
        info.primPath = primPath;
        info.name = primNameFromPath(primPath);
        info.meshData = std::move(meshData);
        assignMaterial(info, boundMaterial(prim), usdPath, result.diagnostics,
                       missingTextureWarnings);
        result.meshes.push_back(std::move(info));
    }

    result.diagnostics.printWarnings("USDLoader " + usdPath);

    return result;
#endif
}

std::vector<USDMeshInfo> USDLoader::loadMeshes(const std::string& usdPath,
                                               float scale) {
    return parse(usdPath, scale).meshes;
}

} // namespace Asset
} // namespace KE

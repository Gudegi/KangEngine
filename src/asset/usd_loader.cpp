#include "usd_loader.hpp"

#include "asset/mesh_loader.hpp"
#include "geometry/mesh_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#ifdef KANGENGINE_USE_USD
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>
#include <pxr/usd/usdPhysics/metrics.h>
#include <pxr/usd/usdPhysics/joint.h>
#include <pxr/usd/usdPhysics/collisionAPI.h>
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

namespace KE {
namespace Asset {
USDArticulationImportResult
USDLoader::parseArticulation(const std::string& usdPath,
                             const std::string& primPath,
                             const std::string& order) {
#ifndef KANGENGINE_USE_USD
    throw std::runtime_error("USD support not compiled");
#else
    using namespace pxr;
    // 1. Open the selected robot subtree and validate units/order supported by
    // ArticulationDesc. No stage-wide unit or axis conversion is performed.
    if (order != "DFS")
        throw std::runtime_error(
            "USD articulation currently supports DFS order only");
    auto stage = UsdStage::Open(usdPath);
    if (!stage)
        throw std::runtime_error("Cannot open USD articulation: " + usdPath);
    if (UsdGeomGetStageUpAxis(stage) != TfToken("Z") ||
        std::abs(UsdGeomGetStageMetersPerUnit(stage) - 1.0) > 1e-9 ||
        std::abs(UsdPhysicsGetStageKilogramsPerUnit(stage) - 1.0) > 1e-9)
        throw std::runtime_error("USD articulation currently requires Z-up and "
                                 "meter units with kilograms");
    auto root = primPath.empty() ? stage->GetDefaultPrim()
                                 : stage->GetPrimAtPath(SdfPath(primPath));
    if (!root)
        throw std::runtime_error("USD articulation root prim is missing");
    // Attribute readers provide defaults for omitted scalar/vector/rotation data.
    auto attr = [](const UsdPrim& p, const char* name) {
        return p.GetAttribute(TfToken(name));
    };
    auto scalar = [&](const UsdPrim& p, const char* name, float fallback) {
        float v = fallback;
        auto a = attr(p, name);
        if (a && a.HasAuthoredValueOpinion() && !a.Get(&v))
            throw std::runtime_error("Invalid USD numeric attribute: " +
                                     a.GetPath().GetString());
        return v;
    };
    auto vec = [&](const UsdPrim& p, const char* name) {
        GfVec3f v(0);
        auto a = attr(p, name);
        if (a)
            a.Get(&v);
        return Eigen::Vector3f(v[0], v[1], v[2]);
    };
    auto quat = [&](const UsdPrim& p, const char* name) {
        GfQuatf v(1);
        auto a = attr(p, name);
        if (a)
            a.Get(&v);
        return Eigen::Quaternionf(v.GetReal(), v.GetImaginary()[0],
                                  v.GetImaginary()[1], v.GetImaginary()[2])
            .normalized();
    };
    // 2. Collect rigid bodies, including bodies inside USD instances.
    UsdGeomXformCache cache;
    std::map<std::string, UsdPrim> bodies, inbound;
    std::map<std::string, std::vector<std::string>> children;
    for (auto p : UsdPrimRange(root, UsdTraverseInstanceProxies())) {
        if (p.HasAPI<UsdPhysicsRigidBodyAPI>())
            bodies[p.GetPath().GetString()] = p;
    }
    // 3. Build the body0 -> body1 joint graph. Each child must have one parent;
    // unsupported joint types and connections outside the selection are rejected.
    for (auto p : UsdPrimRange(root, UsdTraverseInstanceProxies())) {
        if (!p.IsA<UsdPhysicsJoint>())
            continue;
        auto type = p.GetTypeName().GetString();
        if (type != "PhysicsFixedJoint" && type != "PhysicsRevoluteJoint" &&
            type != "PhysicsPrismaticJoint")
            throw std::runtime_error("Unsupported USD joint: " +
                                     p.GetPath().GetString());
        bool enabled = true;
        auto en = attr(p, "physics:jointEnabled");
        if (en)
            en.Get(&enabled);
        if (!enabled)
            throw std::runtime_error(
                "Disabled joints require explicit handling: " +
                p.GetPath().GetString());
        SdfPathVector b0, b1;
        p.GetRelationship(TfToken("physics:body0")).GetTargets(&b0);
        p.GetRelationship(TfToken("physics:body1")).GetTargets(&b1);
        if (b0.size() != 1 || b1.size() != 1 ||
            !bodies.count(b0[0].GetString()) ||
            !bodies.count(b1[0].GetString()))
            throw std::runtime_error(
                "Joint must connect two bodies inside selected USD prim: " +
                p.GetPath().GetString());
        auto child = b1[0].GetString();
        if (inbound.count(child))
            throw std::runtime_error(
                "USD closed chains/multiple inbound joints unsupported");
        inbound[child] = p;
        children[b0[0].GetString()].push_back(child);
    }
    // A free-base tree has exactly one body without an inbound joint.
    std::vector<std::string> roots;
    for (auto& [name, p] : bodies)
        if (!inbound.count(name))
            roots.push_back(name);
    if (roots.size() != 1)
        throw std::runtime_error(
            "Select exactly one connected free-base articulation");

    // 4. Traverse parent-first to assign common indices to the skeleton,
    // joint descriptors, inertials, and the geometry imported below.
    USDArticulationImportResult result;
    auto& out = result.articulation;
    out.assetDir = std::filesystem::path(usdPath).parent_path().string();
    out.traversalOrder = order;
    std::vector<std::string> names;
    std::vector<int> parents, counts;
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Quaternionf> rotations;
    std::map<std::string, int> indices;
    std::function<void(std::string, int)> visit = [&](std::string path,
                                                      int parent) {
        if (indices.count(path))
            throw std::runtime_error("USD joint cycle");
        // Validate each body before assigning its skeleton index.
        auto p = bodies.at(path);
        const auto transform = cache.GetLocalToWorldTransform(p);
        Eigen::Matrix3d basis;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                basis(row, col) = transform[row][col];
        if (!(basis * basis.transpose())
                 .isApprox(Eigen::Matrix3d::Identity(), 1e-5) ||
            basis.determinant() < 0)
            throw std::runtime_error(
                "Scaled or reflected USD rigid body transforms unsupported: " +
                path);
        int i = names.size();
        indices[path] = i;
        bool active = true, kinematic = false;
        UsdPhysicsRigidBodyAPI(p).GetRigidBodyEnabledAttr().Get(&active);
        UsdPhysicsRigidBodyAPI(p).GetKinematicEnabledAttr().Get(&kinematic);
        if (!active || kinematic)
            throw std::runtime_error(
                "USD articulation requires enabled dynamic bodies: " + path);
        std::string name = p.GetName().GetString();
        if (std::find(names.begin(), names.end(), name) != names.end())
            throw std::runtime_error("Duplicate USD body names");
        names.push_back(name);
        parents.push_back(parent);
        Eigen::Vector3f pos = Eigen::Vector3f::Zero();
        Eigen::Quaternionf rot = Eigen::Quaternionf::Identity();
        int n = 0;
        // Derive the zero-position child body pose from the two local joint
        // frames. The root stays at identity; its world pose is caller-owned.
        if (parent >= 0) {
            auto j = inbound.at(path);
            auto q0 = quat(j, "physics:localRot0"),
                 q1 = quat(j, "physics:localRot1");
            auto p1 = vec(j, "physics:localPos1");
            rot = q0 * q1.conjugate();
            pos = vec(j, "physics:localPos0") - rot * p1;
            if (j.GetTypeName() != TfToken("PhysicsFixedJoint")) {
                if (p1.norm() > 1e-7)
                    throw std::runtime_error("Nonzero movable child joint "
                                             "anchor is not representable yet");
                // Fixed joints add no DOF. Movable joints contribute one axis
                // expressed in the child body's frame.
                JointDesc d;
                d.name = j.GetName().GetString();
                bool revolute =
                    j.GetTypeName() == TfToken("PhysicsRevoluteJoint");
                d.type = revolute ? JointDesc::Type::Revolute
                                  : JointDesc::Type::Prismatic;
                TfToken axis("X");
                attr(j, "physics:axis").Get(&axis);
                if (axis != TfToken("X") && axis != TfToken("Y") &&
                    axis != TfToken("Z"))
                    throw std::runtime_error("Invalid USD joint axis");
                Eigen::Vector3f unit =
                    axis == TfToken("X")   ? Eigen::Vector3f::UnitX()
                    : axis == TfToken("Y") ? Eigen::Vector3f::UnitY()
                                           : Eigen::Vector3f::UnitZ();
                d.axis = q1 * unit;
                // USD angular limits/velocities use degrees; engine values use
                // radians. Prismatic values already use the validated meter unit.
                float conversion = revolute ? float(M_PI / 180.) : 1.f;
                d.loLimit =
                    scalar(j, "physics:lowerLimit", -FLT_MAX) * conversion;
                d.hiLimit =
                    scalar(j, "physics:upperLimit", FLT_MAX) * conversion;
                if (d.loLimit > d.hiLimit)
                    throw std::runtime_error("Invalid USD joint limits");
                // Import force-drive gains, effort limits and PhysX joint data;
                // reject drive/friction modes the descriptor cannot represent.
                auto prefix = revolute ? "drive:angular:physics:"
                                       : "drive:linear:physics:";
                TfToken driveType("force");
                auto driveTypeAttr =
                    attr(j, (std::string(prefix) + "type").c_str());
                if (driveTypeAttr)
                    driveTypeAttr.Get(&driveType);
                if (driveType != TfToken("force"))
                    throw std::runtime_error(
                        "USD acceleration drives are not supported");
                if (scalar(j, "physxJoint:jointFriction", 0) != 0)
                    throw std::runtime_error("USD joint friction is not "
                                             "represented by ArticulationDesc");
                d.kp =
                    scalar(j, (std::string(prefix) + "stiffness").c_str(), 0);
                d.kd = scalar(j, (std::string(prefix) + "damping").c_str(), 0);
                d.effortLimit = scalar(
                    j, (std::string(prefix) + "maxForce").c_str(), FLT_MAX);
                d.armature = scalar(j, "physxJoint:armature", 0);
                d.velocityLimit =
                    scalar(j, "physxJoint:maxJointVelocity", FLT_MAX) *
                    conversion;
                out.joints[i].push_back(d);
                n = 1;
            }
        }
        positions.push_back(pos);
        rotations.push_back(rot);
        counts.push_back(n);
        // Import explicit mass, COM and principal inertia in the body frame.
        InertialDesc mass;
        mass.mass = scalar(p, "physics:mass", -1);
        mass.com = vec(p, "physics:centerOfMass");
        mass.quat = quat(p, "physics:principalAxes");
        mass.diagInertia = vec(p, "physics:diagonalInertia");
        // USD sentinel values request geometry-based mass computation. Empty
        // auxiliary links have no geometry from which to derive a COM/frame.
        bool hasCollider = false;
        for (auto descendant : UsdPrimRange(p, UsdTraverseInstanceProxies())) {
            auto owner = descendant;
            while (owner && !owner.HasAPI<UsdPhysicsRigidBodyAPI>())
                owner = owner.GetParent();
            if (owner == p && descendant.HasAPI<UsdPhysicsCollisionAPI>()) {
                bool enabled = true;
                UsdPhysicsCollisionAPI(descendant)
                    .GetCollisionEnabledAttr()
                    .Get(&enabled);
                hasCollider |= enabled;
            }
        }
        if (!mass.com.allFinite() && !hasCollider) {
            mass.com.setZero();
            result.diagnostics.warnings.push_back(
                "USD collider-free body uses origin COM: " + path);
        }
        if (mass.quat.squaredNorm() < 1e-12f &&
            mass.diagInertia.isApprox(
                Eigen::Vector3f::Constant(mass.diagInertia.x()))) {
            mass.quat.setIdentity();
        }
        if (!mass.quat.coeffs().allFinite() || mass.quat.squaredNorm() < 1e-12f)
            throw std::runtime_error(
                "USD automatic principal-axis computation is unsupported: " +
                path);
        if (!std::isfinite(mass.mass) || !mass.diagInertia.allFinite() ||
            !mass.com.allFinite() || mass.mass <= 0 ||
            mass.diagInertia.minCoeff() <= 0)
            throw std::runtime_error(
                "Explicit positive USD mass and inertia required: " + path);
        out.inertials[i] = mass;
        for (auto child : children[path])
            visit(child, i);
    };
    // Finish the skeleton only after every collected body has been reached.
    visit(roots[0], -1);
    if (indices.size() != bodies.size())
        throw std::runtime_error("Disconnected USD bodies");
    out.skeletonTree = std::make_shared<Animation::SkeletonTree>(
        names, parents, positions, rotations, counts);
    // 5. Attach mesh geometry to its nearest rigid-body ancestor. Geometry
    // outside a body is skipped; unsupported colliders fail explicitly.
    for (auto p : UsdPrimRange(root, UsdTraverseInstanceProxies())) {
        auto owner = p;
        while (owner && !owner.HasAPI<UsdPhysicsRigidBodyAPI>())
            owner = owner.GetParent();
        if (!owner)
            continue;
        bool collision = false;
        if (p.HasAPI<UsdPhysicsCollisionAPI>())
            UsdPhysicsCollisionAPI(p).GetCollisionEnabledAttr().Get(&collision);
        if (!p.IsA<UsdGeomMesh>()) {
            if (collision)
                throw std::runtime_error(
                    "USD articulation collider type unsupported: " +
                    p.GetPath().GetString());
            if (p.IsA<UsdGeomGprim>())
                result.diagnostics.warnings.push_back(
                    "Unsupported USD visual geometry: " +
                    p.GetPath().GetString());
            continue;
        }
        int i = indices.at(owner.GetPath().GetString());
        // loadMeshData returns world geometry; undo the authored body pose.
        auto mesh = loadMeshData(p, nullptr);
        auto inv = cache.GetLocalToWorldTransform(owner).GetInverse();
        auto normal = inv.GetInverse().GetTranspose();
        for (auto& v : mesh.vertices) {
            auto w = inv.Transform(GfVec3d(v.x, v.y, v.z));
            v = glm::vec3(w[0], w[1], w[2]);
        }
        for (auto& n : mesh.normals) {
            auto w = normal.TransformDir(GfVec3d(n.x, n.y, n.z));
            n = glm::normalize(glm::vec3(w[0], w[1], w[2]));
        }
        if (mesh.vertices.empty() || mesh.indices.empty())
            throw std::runtime_error("Empty USD articulation mesh: " +
                                     p.GetPath().GetString());
        // The same body-local mesh can serve collision and visualization.
        // Only authored convexHull colliders are accepted; no decomposition is done.
        auto data = std::make_shared<Scene::MeshData>(std::move(mesh));
        if (collision) {
            TfToken approximation;
            attr(p, "physics:approximation").Get(&approximation);
            if (approximation != TfToken("convexHull"))
                throw std::runtime_error("USD articulation requires explicit "
                                         "convexHull mesh collision");
            CollisionGeomDesc d;
            d.name = p.GetName().GetString();
            d.type = CollisionGeomDesc::Type::ConvexMesh;
            d.meshData = data;
            d.margin = scalar(p, "physxCollision:contactOffset", -1);
            out.collisionGeoms[i].push_back(d);
        }
        // Visual geometry follows USD visibility independently of collision.
        if (UsdGeomImageable(p).ComputeVisibility() !=
            UsdGeomTokens->invisible) {
            VisualGeomDesc d;
            d.bodyName = names[i];
            d.bodyIndex = i;
            d.meshData = data;
            out.visualGeoms.push_back(d);
        }
    }
    // 6. Report the settings left to the caller instead of silently implying
    // that the imported descriptor reproduces every USD simulation property.
    result.diagnostics.warnings.push_back(
        "USD articulation: root pose, initial joint state, drive targets, "
        "damping, solver settings and self-collision are supplied by the "
        "caller. Physics material bindings, rest offsets, collision filtering "
        "and mesh material appearance are not imported.");
    return result;
#endif
}
} // namespace Asset
} // namespace KE

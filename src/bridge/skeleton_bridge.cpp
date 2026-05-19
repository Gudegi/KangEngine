#include "skeleton_bridge.hpp"
#include "asset/mjcf_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/load_utils.hpp"
#include "utils/types.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fmt/core.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace KE {
namespace Bridge {

namespace fs = std::filesystem;

// Build a merged MeshData from all collision geoms of one body,
// with each geom's vertices/normals transformed into body-local space.
static Scene::MeshData
buildCollisionMesh(const std::vector<Animation::CollisionGeom>& geoms) {
    Scene::MeshData combined;
    for (const auto& geom : geoms) {
        glm::vec3 localPos;
        glm::quat localQuat;
        Scene::MeshData part;

        if (geom.hasFromTo) {
            Eigen::Vector3f center = (geom.from + geom.to) * 0.5f;
            Eigen::Vector3f axis = (geom.to - geom.from).normalized();
            float halfLen = (geom.to - geom.from).norm() * 0.5f;
            Eigen::Quaternionf eq = Eigen::Quaternionf::FromTwoVectors(
                Eigen::Vector3f::UnitZ(), axis);
            localPos = glm::vec3(center.x(), center.y(), center.z());
            localQuat = glm::quat(eq.w(), eq.x(), eq.y(), eq.z());
            float r = geom.size[0];
            if (geom.type == Animation::CollisionGeom::Type::Capsule)
                part = Scene::Prim::createCapsuleData(r, halfLen * 2.f,
                                                      UpAxis::Z, 12);
            else
                part = Scene::Prim::createCylinderData(r, halfLen * 2.f,
                                                       UpAxis::Z, 12);
        } else {
            localPos = glm::vec3(geom.pos.x(), geom.pos.y(), geom.pos.z());
            localQuat = glm::quat(geom.quat.w(), geom.quat.x(), geom.quat.y(),
                                  geom.quat.z());
            switch (geom.type) {
            case Animation::CollisionGeom::Type::Sphere:
                part = Scene::Prim::createSphereData(geom.size[0], 12, 8);
                break;
            case Animation::CollisionGeom::Type::Capsule:
                part = Scene::Prim::createCapsuleData(
                    geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                break;
            case Animation::CollisionGeom::Type::Cylinder:
                part = Scene::Prim::createCylinderData(
                    geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                break;
            case Animation::CollisionGeom::Type::Box:
                part = Scene::Prim::createRectangleData(
                    geom.size[0] * 2.f, geom.size[1] * 2.f, geom.size[2] * 2.f);
                break;
            }
        }

        // Transform vertices/normals into body-local space
        for (auto& v : part.vertices)
            v = localQuat * v + localPos;
        for (auto& n : part.normals)
            n = localQuat * n;

        // Merge into combined mesh
        unsigned int offset = (unsigned int)combined.vertices.size();
        for (auto& idx : part.indices)
            idx += offset;
        combined.vertices.insert(combined.vertices.end(), part.vertices.begin(),
                                 part.vertices.end());
        combined.normals.insert(combined.normals.end(), part.normals.begin(),
                                part.normals.end());
        combined.indices.insert(combined.indices.end(), part.indices.begin(),
                                part.indices.end());
    }
    return combined;
}

static std::string lowerExtension(std::string path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos)
        return "";
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

static Scene::MeshData loadVisualMesh(const std::string& path) {
    std::string ext = lowerExtension(path);
    if (ext == ".stl")
        return loadStl(path);
    if (ext == ".obj")
        return loadObj(path);
    throw std::runtime_error("Unsupported visual mesh extension: " + path);
}

static void applyMeshInfoTransform(Scene::MeshData& mesh,
                                   const Animation::MeshInfo& meshInfo) {
    glm::vec3 localPos(meshInfo.pos.x(), meshInfo.pos.y(), meshInfo.pos.z());
    glm::quat localQuat(meshInfo.quat.w(), meshInfo.quat.x(), meshInfo.quat.y(),
                        meshInfo.quat.z());
    for (auto& v : mesh.vertices)
        v = localQuat * v + localPos;
    for (auto& n : mesh.normals)
        n = localQuat * n;
}

static void appendMesh(Scene::MeshData& dst, Scene::MeshData&& part) {
    unsigned int offset = static_cast<unsigned int>(dst.vertices.size());
    for (auto& idx : part.indices)
        idx += offset;
    dst.vertices.insert(dst.vertices.end(), part.vertices.begin(),
                        part.vertices.end());
    dst.normals.insert(dst.normals.end(), part.normals.begin(),
                       part.normals.end());
    dst.uvs.insert(dst.uvs.end(), part.uvs.begin(), part.uvs.end());
    dst.indices.insert(dst.indices.end(), part.indices.begin(),
                       part.indices.end());
}

static std::vector<std::shared_ptr<Scene::MeshData>>
buildBodyMeshes(const Animation::CharacterData& data) {
    int numBodies = data.skeletonTree ? data.skeletonTree->numJoints() : 0;
    std::vector<std::shared_ptr<Scene::MeshData>> bodyMeshes(numBodies);

    std::unordered_set<int> stlBodies;
    for (const auto& m : data.meshInfos)
        stlBodies.insert(m.bodyIndex);

    for (int i = 0; i < numBodies; i++) {
        if (stlBodies.count(i) != 0)
            continue;
        auto it = data.collisionGeoms.find(i);
        if (it == data.collisionGeoms.end() || it->second.empty())
            continue;
        auto colMesh = buildCollisionMesh(it->second);
        if (!colMesh.vertices.empty())
            bodyMeshes[i] =
                std::make_shared<Scene::MeshData>(std::move(colMesh));
    }

    std::unordered_map<int, Scene::MeshData> visualMeshes;
    for (const auto& meshInfo : data.meshInfos) {
        std::string meshPath = (fs::path(data.meshDir) / meshInfo.meshFile)
                                   .lexically_normal()
                                   .string();
        fmt::print("Loading mesh [{}] {}: {}\n", meshInfo.bodyIndex,
                   meshInfo.bodyName, meshPath);
        auto part = loadVisualMesh(meshPath);
        applyMeshInfoTransform(part, meshInfo);
        auto it = visualMeshes.find(meshInfo.bodyIndex);
        if (it == visualMeshes.end()) {
            visualMeshes[meshInfo.bodyIndex] = std::move(part);
        } else {
            appendMesh(it->second, std::move(part));
        }
    }

    for (auto& [bodyIdx, mesh] : visualMeshes) {
        if (bodyIdx >= 0 && bodyIdx < static_cast<int>(bodyMeshes.size()))
            bodyMeshes[bodyIdx] =
                std::make_shared<Scene::MeshData>(std::move(mesh));
    }

    return bodyMeshes;
}

SkeletonBridge SkeletonBridge::fromMJCF(const std::string& mjcfPath,
                                        Scene::SceneBackend* scene,
                                        const std::string& primBasePath,
                                        float scale, const std::string& order,
                                        const std::string& meshAssetBasePath) {
    auto asset = SkeletonBridgeAsset::fromMJCF(mjcfPath, scale, order);
    return asset.instantiate(scene, primBasePath, meshAssetBasePath);
}

SkeletonBridge SkeletonBridge::fromData(const Animation::CharacterData& data,
                                        Scene::SceneBackend* scene,
                                        const std::string& primBasePath,
                                        float scale,
                                        const std::string& meshAssetBasePath) {
    auto asset = SkeletonBridgeAsset::fromData(data, scale);
    return asset.instantiate(scene, primBasePath, meshAssetBasePath);
}

SkeletonBridgeAsset SkeletonBridgeAsset::fromMJCF(const std::string& mjcfPath,
                                                  float scale,
                                                  const std::string& order) {
    return fromData(Animation::MJCFLoader::load(mjcfPath, 1.0f, order), scale);
}

SkeletonBridgeAsset
SkeletonBridgeAsset::fromData(const Animation::CharacterData& data,
                              float scale) {
    SkeletonBridgeAsset asset;
    asset._data = data;
    asset._scale = scale;
    asset._bodyMeshes = buildBodyMeshes(data);
    fmt::print("SkeletonBridgeAsset loaded: {} bodies, {} meshes\n",
               asset.numBodies(), data.meshInfos.size());
    return asset;
}

void SkeletonBridgeAsset::defineMeshAssets(
    Scene::SceneBackend* scene, const std::string& meshAssetBasePath) const {
    if (!scene || meshAssetBasePath.empty())
        return;
    for (int i = 0; i < static_cast<int>(_bodyMeshes.size()); i++) {
        if (!_bodyMeshes[i])
            continue;
        auto* assetPrim =
            scene->definePrim(meshAssetBasePath + "/body_" + std::to_string(i),
                              Scene::PrimType::Mesh);
        if (!assetPrim->getMeshData())
            assetPrim->setMeshData(_bodyMeshes[i]);
    }
}

SkeletonBridge
SkeletonBridgeAsset::instantiate(Scene::SceneBackend* scene,
                                 const std::string& primBasePath,
                                 const std::string& meshAssetBasePath) const {
    SkeletonBridge bridge;
    bridge._fk = Animation::SkeletonFK::fromData(_data, _scale);

    auto globalTransforms = bridge._fk.state().computeGlobalTransforms();
    int numBodies = bridge._fk.numBodies();
    bridge._bodyPrims.resize(numBodies, nullptr);
    const bool useMeshInstances = !meshAssetBasePath.empty();
    if (useMeshInstances)
        defineMeshAssets(scene, meshAssetBasePath);

    // Create one Prim per body
    for (int i = 0; i < numBodies; i++) {
        std::string bodyName = bridge._fk.skeleton().nodeName(i);
        std::string primPath = primBasePath + "/" + bodyName;
        std::string meshSourcePath =
            meshAssetBasePath + "/body_" + std::to_string(i);
        auto* prim = scene->definePrim(
            primPath, useMeshInstances ? Scene::PrimType::MeshInstance
                                       : Scene::PrimType::Mesh);
        if (useMeshInstances)
            prim->setMeshSourcePath(meshSourcePath);
        prim->setAttribute("primvars:displaycolorAlpha",
                           glm::vec4(0.15f, 0.15f, 0.15f, 1.0f));
        prim->setAttribute("xformOp:scale", glm::vec3(_scale));
        glm::vec3 pos =
            Animation::toGlm(globalTransforms[i].translation) * _scale;
        glm::quat rot = Animation::toGlm(globalTransforms[i].rotation);
        prim->setAttribute("xformOp:translate", pos);
        prim->setAttribute("xformOp:rotateQuaternion", rot);
        bridge._bodyPrims[i] = prim;

        if (!useMeshInstances && i < static_cast<int>(_bodyMeshes.size()) &&
            _bodyMeshes[i])
            prim->setMeshData(_bodyMeshes[i]);
    }

    fmt::print("SkeletonBridge instantiated: {} bodies\n",
               bridge._fk.numBodies());
    return bridge;
}

void SkeletonBridge::applyPose() {
    auto globals = _fk.state().computeGlobalTransforms();
    float scale = _fk.scale();
    for (int i = 0; i < _fk.numBodies(); ++i) {
        if (!_bodyPrims[i])
            continue;
        glm::vec3 pos = Animation::toGlm(globals[i].translation) * scale;
        glm::quat rot = Animation::toGlm(globals[i].rotation);
        _bodyPrims[i]->setAttribute("xformOp:translate", pos);
        _bodyPrims[i]->setAttribute("xformOp:rotateQuaternion", rot);
    }
}

void SkeletonBridge::setJointRotation(int idx, const Eigen::Quaternionf& q) {
    _fk.setJointRotation(idx, q);
}

void SkeletonBridge::setRootTranslation(const Eigen::Vector3f& t) {
    _fk.setRootTranslation(t);
}

void SkeletonBridge::resetToZeroPose() {
    _fk.resetToZeroPose();
    applyPose();
}

} // namespace Bridge
} // namespace KE

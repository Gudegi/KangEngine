#include "skeleton_bridge.hpp"
#include "asset/mesh_loader.hpp"
#include "asset/mjcf_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "engine/scene/component/articulation_component.hpp"
#include "engine/scene/component/articulation_binding_component.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/types.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fmt/core.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
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
        return Asset::loadStl(path);
    if (ext == ".obj")
        return Asset::loadObj(path);
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

static std::vector<SkeletonBridgeAsset::VisualGeomAsset>
buildVisualGeomAssets(const Animation::CharacterData& data) {
    std::vector<SkeletonBridgeAsset::VisualGeomAsset> visualGeomAssets;
    visualGeomAssets.reserve(data.meshInfos.size());

    for (const auto& meshInfo : data.meshInfos) {
        std::string meshPath = (fs::path(data.meshDir) / meshInfo.meshFile)
                                   .lexically_normal()
                                   .string();
        fmt::print("Loading mesh [{}] {}: {}\n", meshInfo.bodyIndex,
                   meshInfo.bodyName, meshPath);
        auto part = loadVisualMesh(meshPath);
        applyMeshInfoTransform(part, meshInfo);

        SkeletonBridgeAsset::VisualGeomAsset asset;
        asset.bodyIndex = meshInfo.bodyIndex;
        asset.mesh = std::make_shared<Scene::MeshData>(std::move(part));
        asset.color = meshInfo.rgba;
        visualGeomAssets.emplace_back(std::move(asset));
    }

    return visualGeomAssets;
}

static std::vector<std::shared_ptr<Scene::MeshData>> buildBodyMeshesFromVisuals(
    int numBodies,
    const std::vector<SkeletonBridgeAsset::VisualGeomAsset>& visualGeomAssets,
    const Animation::CollisionGeomMap& collisionGeoms) {
    std::vector<std::shared_ptr<Scene::MeshData>> bodyMeshes(numBodies);
    std::vector<bool> hasVisual(static_cast<size_t>(numBodies), false);

    std::unordered_map<int, Scene::MeshData> mergedBodyVisualMeshes;
    for (const auto& visual : visualGeomAssets) {
        const int bodyIdx = visual.bodyIndex;
        if (bodyIdx < 0 || bodyIdx >= numBodies || !visual.mesh)
            continue;
        hasVisual[static_cast<size_t>(bodyIdx)] = true;
        auto part = *visual.mesh;
        auto it = mergedBodyVisualMeshes.find(bodyIdx);
        if (it == mergedBodyVisualMeshes.end()) {
            mergedBodyVisualMeshes[bodyIdx] = std::move(part);
        } else {
            appendMesh(it->second, std::move(part));
        }
    }

    for (auto& [bodyIdx, mesh] : mergedBodyVisualMeshes) {
        if (bodyIdx >= 0 && bodyIdx < static_cast<int>(bodyMeshes.size()))
            bodyMeshes[bodyIdx] =
                std::make_shared<Scene::MeshData>(std::move(mesh));
    }

    for (int i = 0; i < numBodies; i++) {
        if (hasVisual[static_cast<size_t>(i)])
            continue;
        auto it = collisionGeoms.find(i);
        if (it == collisionGeoms.end() || it->second.empty())
            continue;
        auto colMesh = buildCollisionMesh(it->second);
        if (!colMesh.vertices.empty())
            bodyMeshes[i] =
                std::make_shared<Scene::MeshData>(std::move(colMesh));
    }

    return bodyMeshes;
}

static std::vector<Eigen::Vector4f> buildBodyColorsFromVisuals(
    int numBodies,
    const std::vector<SkeletonBridgeAsset::VisualGeomAsset>& visualGeomAssets) {
    std::vector<Eigen::Vector4f> colors(
        numBodies, Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f));
    std::vector<bool> assigned(static_cast<size_t>(numBodies), false);

    for (const auto& visual : visualGeomAssets) {
        const int bodyIdx = visual.bodyIndex;
        if (bodyIdx < 0 || bodyIdx >= numBodies)
            continue;
        if (assigned[static_cast<size_t>(bodyIdx)])
            continue;
        colors[static_cast<size_t>(bodyIdx)] = visual.color;
        assigned[static_cast<size_t>(bodyIdx)] = true;
    }

    return colors;
}

static std::vector<bool> buildBodyHasVisualMap(
    int numBodies,
    const std::vector<SkeletonBridgeAsset::VisualGeomAsset>& visualGeomAssets) {
    std::vector<bool> hasVisual(static_cast<size_t>(numBodies), false);
    for (const auto& visual : visualGeomAssets) {
        if (visual.bodyIndex >= 0 && visual.bodyIndex < numBodies &&
            visual.mesh) {
            hasVisual[static_cast<size_t>(visual.bodyIndex)] = true;
        }
    }
    return hasVisual;
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
    auto asset =
        fromData(Asset::MJCFLoader::load(mjcfPath, 1.0f, order), scale);
    asset._assetPath = mjcfPath;
    return asset;
}

SkeletonBridgeAsset
SkeletonBridgeAsset::fromData(const Animation::CharacterData& data,
                              float scale) {
    SkeletonBridgeAsset asset;
    asset._data = data;
    asset._scale = scale;
    asset._visualGeomAssets = buildVisualGeomAssets(data);
    fmt::print("SkeletonBridgeAsset loaded: {} bodies, {} meshes\n",
               asset.numBodies(), data.meshInfos.size());
    return asset;
}

void SkeletonBridgeAsset::defineMeshAssets(Scene::SceneBackend* scene,
                                           const std::string& meshAssetBasePath,
                                           bool splitVisualGeoms) const {
    if (!scene || meshAssetBasePath.empty())
        return;
    const int bodies = numBodies();
    auto bodyHasVisual = buildBodyHasVisualMap(bodies, _visualGeomAssets);

    if (!splitVisualGeoms) {
        auto bodyMeshes = buildBodyMeshesFromVisuals(bodies, _visualGeomAssets,
                                                     _data.collisionGeoms);
        auto bodyColors = buildBodyColorsFromVisuals(bodies, _visualGeomAssets);
        for (int i = 0; i < static_cast<int>(bodyMeshes.size()); i++) {
            if (!bodyMeshes[i])
                continue;
            auto* assetPrim = scene->definePrim(meshAssetBasePath + "/body_" +
                                                    std::to_string(i),
                                                Scene::PrimType::Mesh);
            if (!assetPrim->getMeshData())
                assetPrim->setMeshData(bodyMeshes[i]);
            if (i < static_cast<int>(bodyColors.size())) {
                const auto& c = bodyColors[static_cast<size_t>(i)];
                assetPrim->setDisplayColorAlpha(
                    glm::vec4(c.x(), c.y(), c.z(), c.w()));
            }
        }
        return;
    }

    for (int i = 0; i < static_cast<int>(_visualGeomAssets.size()); i++) {
        const auto& visual = _visualGeomAssets[static_cast<size_t>(i)];
        if (!visual.mesh)
            continue;
        auto* assetPrim = scene->definePrim(meshAssetBasePath + "/visual_" +
                                                std::to_string(i),
                                            Scene::PrimType::Mesh);
        if (!assetPrim->getMeshData())
            assetPrim->setMeshData(visual.mesh);
        const auto& c = visual.color;
        assetPrim->setDisplayColorAlpha(glm::vec4(c.x(), c.y(), c.z(), c.w()));
    }

    auto bodyMeshes = buildBodyMeshesFromVisuals(bodies, _visualGeomAssets,
                                                 _data.collisionGeoms);
    auto bodyColors = buildBodyColorsFromVisuals(bodies, _visualGeomAssets);
    for (int i = 0; i < static_cast<int>(bodyMeshes.size()); i++) {
        if (!bodyMeshes[i] || bodyHasVisual[static_cast<size_t>(i)])
            continue;
        auto* assetPrim =
            scene->definePrim(meshAssetBasePath + "/body_" + std::to_string(i),
                              Scene::PrimType::Mesh);
        if (!assetPrim->getMeshData())
            assetPrim->setMeshData(bodyMeshes[i]);
        if (i < static_cast<int>(bodyColors.size())) {
            const auto& c = bodyColors[static_cast<size_t>(i)];
            assetPrim->setDisplayColorAlpha(
                glm::vec4(c.x(), c.y(), c.z(), c.w()));
        }
    }
}

SkeletonBridge SkeletonBridgeAsset::instantiate(
    Scene::SceneBackend* scene, const std::string& primBasePath,
    const std::string& meshAssetBasePath, bool splitVisualGeoms) const {
    SkeletonBridge bridge;
    bridge._fk = Animation::SkeletonFK::fromData(_data, _scale);

    auto globalTransforms = bridge._fk.state().computeGlobalTransforms();
    int numBodies = bridge._fk.numBodies();
    bridge._bodyPrims.resize(numBodies, nullptr);
    auto* rootPrim = scene->definePrim(primBasePath, Scene::PrimType::Xform);
    const bool useMeshInstances = !meshAssetBasePath.empty();
    std::vector<std::shared_ptr<Scene::MeshData>> bodyMeshes;
    if (!useMeshInstances) {
        bodyMeshes = buildBodyMeshesFromVisuals(numBodies, _visualGeomAssets,
                                                _data.collisionGeoms);
    }
    auto bodyColors = buildBodyColorsFromVisuals(numBodies, _visualGeomAssets);
    auto bodyHasVisual = buildBodyHasVisualMap(numBodies, _visualGeomAssets);
    if (useMeshInstances)
        defineMeshAssets(scene, meshAssetBasePath, splitVisualGeoms);

    // Create one Prim per body
    for (int i = 0; i < numBodies; i++) {
        std::string bodyName = bridge._fk.skeleton().nodeName(i);
        std::string primPath = primBasePath + "/" + bodyName;
        std::string meshSourcePath =
            meshAssetBasePath + "/body_" + std::to_string(i);
        const bool hasBodyMesh =
            useMeshInstances || (i < static_cast<int>(bodyMeshes.size()) &&
                                 bodyMeshes[i] != nullptr);
        const bool hasSplitVisual =
            i < static_cast<int>(bodyHasVisual.size()) &&
            bodyHasVisual[static_cast<size_t>(i)];
        const bool bodyIsRenderable = !splitVisualGeoms || !hasSplitVisual;
        auto* prim = scene->definePrim(
            primPath, bodyIsRenderable
                          ? (useMeshInstances ? Scene::PrimType::MeshInstance
                                              : Scene::PrimType::Mesh)
                          : Scene::PrimType::Xform);
        if (bodyIsRenderable && useMeshInstances)
            prim->setMeshSourcePath(meshSourcePath);
        glm::vec4 displayColor(0.15f, 0.15f, 0.15f, 1.0f);
        if (i < static_cast<int>(bodyColors.size())) {
            const auto& c = bodyColors[static_cast<size_t>(i)];
            displayColor = glm::vec4(c.x(), c.y(), c.z(), c.w());
        }
        prim->setDisplayColorAlpha(displayColor);
        prim->setAttribute("xformOp:scale", glm::vec3(_scale));
        glm::vec3 pos =
            Animation::toGlm(globalTransforms[i].translation) * _scale;
        glm::quat rot = Animation::toGlm(globalTransforms[i].rotation);
        prim->setWorldMatrix(glm::translate(glm::mat4(1.0f), pos) *
                             glm::mat4_cast(rot));
        prim->addArticulationBindingComponent()->setBinding(
            Scene::ArticulationPrimRole::BodyFrame, i, bodyName, primBasePath);
        bridge._bodyPrims[i] = prim;

        if (bodyIsRenderable && !useMeshInstances && hasBodyMesh)
            prim->setMeshData(bodyMeshes[i]);
        if (bodyIsRenderable) {
            bridge._renderPrims.push_back(prim);
            bridge._renderPrimBodyIndices.push_back(i);
        }
    }

    if (splitVisualGeoms) {
        std::vector<int> visualCounts(static_cast<size_t>(numBodies), 0);
        for (int visualIndex = 0;
             visualIndex < static_cast<int>(_visualGeomAssets.size());
             visualIndex++) {
            const auto& visual =
                _visualGeomAssets[static_cast<size_t>(visualIndex)];
            const int bodyIndex = visual.bodyIndex;
            if (bodyIndex < 0 || bodyIndex >= numBodies ||
                !bridge._bodyPrims[static_cast<size_t>(bodyIndex)] ||
                !visual.mesh)
                continue;

            const int localIndex =
                visualCounts[static_cast<size_t>(bodyIndex)]++;
            auto* bodyPrim = bridge._bodyPrims[static_cast<size_t>(bodyIndex)];
            const std::string bodyName =
                bridge._fk.skeleton().nodeName(bodyIndex);
            auto* visualPrim = scene->definePrim(
                bodyPrim->getPath() + "/visual_" + std::to_string(localIndex),
                useMeshInstances ? Scene::PrimType::MeshInstance
                                 : Scene::PrimType::Mesh);
            if (useMeshInstances) {
                visualPrim->setMeshSourcePath(meshAssetBasePath + "/visual_" +
                                              std::to_string(visualIndex));
            } else {
                visualPrim->setMeshData(visual.mesh);
            }
            const auto& c = visual.color;
            visualPrim->setDisplayColorAlpha(
                glm::vec4(c.x(), c.y(), c.z(), c.w()));
            visualPrim->addArticulationBindingComponent()->setBinding(
                Scene::ArticulationPrimRole::VisualGeom, bodyIndex, bodyName,
                primBasePath);
            bridge._renderPrims.push_back(visualPrim);
            bridge._renderPrimBodyIndices.push_back(bodyIndex);
        }
    }

    auto articulationComponent = rootPrim->getArticulationComponent();
    if (!articulationComponent)
        articulationComponent = rootPrim->addArticulationComponent();
    articulationComponent->setArticulationMetadata(
        primBasePath, _assetPath, meshAssetBasePath, numBodies,
        static_cast<int>(bridge._renderPrims.size()), splitVisualGeoms);

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
        _bodyPrims[i]->setWorldMatrix(glm::translate(glm::mat4(1.0f), pos) *
                                      glm::mat4_cast(rot));
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

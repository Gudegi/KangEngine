#include "skeleton_bridge.hpp"
#include "animation/mjcf_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/load_utils.hpp"
#include "utils/types.hpp"

#include <Eigen/Geometry>
#include <fmt/core.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace KE {
namespace Bridge {

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

SkeletonBridge SkeletonBridge::fromMJCF(const std::string& mjcfPath,
                                        Scene::SceneBackend* scene,
                                        const std::string& primBasePath,
                                        float scale, const std::string& order) {
    return fromData(Animation::MJCFLoader::load(mjcfPath, 1.0f, order), scene,
                    primBasePath, scale);
}

SkeletonBridge SkeletonBridge::fromData(const Animation::CharacterData& data,
                                        Scene::SceneBackend* scene,
                                        const std::string& primBasePath,
                                        float scale) {
    SkeletonBridge bridge;
    bridge._fk = Animation::SkeletonFK::fromData(data, scale);

    auto globalTransforms = bridge._fk.state().computeGlobalTransforms();
    int numBodies = bridge._fk.numBodies();
    bridge._bodyPrims.resize(numBodies, nullptr);

    // Collect which bodies have STL visual meshes
    std::unordered_set<int> stlBodies;
    for (const auto& m : data.meshInfos)
        stlBodies.insert(m.bodyIndex);

    // Create one Prim per body
    for (int i = 0; i < numBodies; i++) {
        std::string bodyName = bridge._fk.skeleton().nodeName(i);
        std::string primPath = primBasePath + "/" + bodyName;
        auto* prim = scene->definePrim(primPath, Scene::PrimType::Mesh);
        prim->setAttribute("primvars:displaycolorAlpha",
                           glm::vec4(0.15f, 0.15f, 0.15f, 1.0f));
        prim->setAttribute("xformOp:scale", glm::vec3(scale));
        glm::vec3 pos =
            Animation::toGlm(globalTransforms[i].translation) * scale;
        glm::quat rot = Animation::toGlm(globalTransforms[i].rotation);
        prim->setAttribute("xformOp:translate", pos);
        prim->setAttribute("xformOp:rotateQuaternion", rot);
        bridge._bodyPrims[i] = prim;

        // Fallback: build mesh from collision geoms if no STL
        if (stlBodies.count(i) == 0) {
            auto it = data.collisionGeoms.find(i);
            if (it != data.collisionGeoms.end() && !it->second.empty()) {
                auto colMesh = buildCollisionMesh(it->second);
                if (!colMesh.vertices.empty())
                    prim->setMeshData(
                        std::make_shared<Scene::MeshData>(std::move(colMesh)));
            }
        }
    }

    // Load STL mesh data, merging multiple visual geoms per body
    std::unordered_map<int, Scene::MeshData> bodyMeshes;
    for (const auto& meshInfo : data.meshInfos) {
        std::string stlPath = data.meshDir + meshInfo.meshFile;
        fmt::print("Loading mesh [{}] {}: {}\n", meshInfo.bodyIndex,
                   meshInfo.bodyName, stlPath);
        auto part = loadStl(stlPath);
        auto it = bodyMeshes.find(meshInfo.bodyIndex);
        if (it == bodyMeshes.end()) {
            bodyMeshes[meshInfo.bodyIndex] = std::move(part);
        } else {
            auto& dst = it->second;
            unsigned int offset = (unsigned int)dst.vertices.size();
            for (auto& idx : part.indices)
                idx += offset;
            dst.vertices.insert(dst.vertices.end(), part.vertices.begin(),
                                part.vertices.end());
            dst.normals.insert(dst.normals.end(), part.normals.begin(),
                               part.normals.end());
            dst.indices.insert(dst.indices.end(), part.indices.begin(),
                               part.indices.end());
        }
    }
    for (auto& [bodyIdx, mesh] : bodyMeshes)
        bridge._bodyPrims[bodyIdx]->setMeshData(
            std::make_shared<Scene::MeshData>(std::move(mesh)));

    fmt::print("SkeletonBridge loaded: {} bodies, {} meshes\n",
               bridge._fk.numBodies(), data.meshInfos.size());
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

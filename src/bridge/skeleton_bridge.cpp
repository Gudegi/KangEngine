#include "skeleton_bridge.hpp"
#include "animation/mjcf_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/native/prim.hpp"
#include "utils/load_utils.hpp"

#include <fmt/core.h>

namespace KE {
namespace Bridge {

SkeletonBridge SkeletonBridge::fromMJCF(const std::string& mjcfPath,
                                        Scene::SceneBackend* scene,
                                        const std::string& primBasePath,
                                        float scale, const std::string& order) {
    return fromData(Animation::MJCFLoader::load(mjcfPath, 1.0f, order),
                    scene, primBasePath, scale);
}

SkeletonBridge SkeletonBridge::fromData(const Animation::MJCFData& data,
                                        Scene::SceneBackend* scene,
                                        const std::string& primBasePath,
                                        float scale) {
    SkeletonBridge bridge;
    bridge._fk = Animation::SkeletonFK::fromData(data, scale);

    auto globalTransforms = bridge._fk.state().computeGlobalTransforms();
    bridge._bodyPrims.resize(bridge._fk.numBodies(), nullptr);

    for (const auto& meshInfo : data.meshInfos) {
        if (bridge._bodyPrims[meshInfo.bodyIndex] != nullptr) {
            fmt::print("Skipping duplicate mesh [{}] {}: {}\n",
                       meshInfo.bodyIndex, meshInfo.bodyName,
                       meshInfo.meshFile);
            continue;
        }

        std::string stlPath = data.meshDir + meshInfo.meshFile;
        fmt::print("Loading mesh [{}] {}: {}\n", meshInfo.bodyIndex,
                   meshInfo.bodyName, stlPath);

        auto meshData = std::make_shared<Scene::MeshData>(loadStl(stlPath));

        std::string primPath = primBasePath + "/" + meshInfo.bodyName;
        auto* prim = scene->definePrim(primPath, Scene::PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->setAttribute("primvars:displaycolorAlpha",
                           glm::vec4(0.15f, 0.15f, 0.15f, 1.0f));
        prim->setAttribute("xformOp:scale", glm::vec3(scale));

        int idx = meshInfo.bodyIndex;
        glm::vec3 pos =
            Animation::toGlm(globalTransforms[idx].translation) * scale;
        glm::quat rot = Animation::toGlm(globalTransforms[idx].rotation);
        prim->setAttribute("xformOp:translate", pos);
        prim->setAttribute("xformOp:rotateQuaternion", rot);

        bridge._bodyPrims[idx] = prim;
    }

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

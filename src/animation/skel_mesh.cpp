#include "skel_mesh.hpp"
#include "skeleton_math.hpp"
#include "utils/load_utils.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/native/prim.hpp"

#include <fmt/core.h>

namespace KE {
namespace Animation {

SkelMesh SkelMesh::fromMJCF(const std::string& mjcfPath,
                            Scene::SceneBackend* scene,
                            const std::string& primBasePath, float scale,
                            const std::string& order) {
    SkelMesh skelMesh;
    skelMesh._scale = scale;

    // 1. Load MJCF
    auto mjcfData = MJCFLoader::loadSkelMesh(mjcfPath, order);
    skelMesh._skeleton =
        std::make_shared<const SkeletonTree>(std::move(mjcfData.skeleton));
    skelMesh._skeleton->print();

    // 2. Zero pose (rest pose)
    skelMesh._state = SkeletonState::zeroPose(skelMesh._skeleton);
    auto globalTransforms = skelMesh._state.computeGlobalTransforms();
    skelMesh._state.printGlobalPositions();

    // 3. Create prims per body with STL meshes
    skelMesh._bodyPrims.resize(skelMesh._skeleton->numJoints(), nullptr);

    for (const auto& meshInfo : mjcfData.meshInfos) {
        // Skip if this body already has a mesh (e.g. logo_link on torso)
        if (skelMesh._bodyPrims[meshInfo.bodyIndex] != nullptr) {
            fmt::print("Skipping duplicate mesh [{}] {}: {}\n",
                       meshInfo.bodyIndex, meshInfo.bodyName,
                       meshInfo.meshFile);
            continue;
        }

        std::string stlPath = mjcfData.meshDir + meshInfo.meshFile;
        fmt::print("Loading mesh [{}] {}: {}\n", meshInfo.bodyIndex,
                   meshInfo.bodyName, stlPath);

        auto meshData = std::make_shared<Scene::MeshData>(loadStl(stlPath));

        std::string primPath = primBasePath + "/" + meshInfo.bodyName;
        auto* prim = scene->definePrim(primPath, Scene::PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->setAttribute("primvars:displaycolorAlpha",
                           glm::vec4(0.15f, 0.15f, 0.15f, 1.0f));
        prim->setAttribute("xformOp:scale", glm::vec3(scale));

        // Set global transform from FK
        int idx = meshInfo.bodyIndex;
        const auto& tf = globalTransforms[idx];
        glm::vec3 pos = toGlm(tf.translation) * scale;
        glm::quat rot = toGlm(tf.rotation);
        prim->setAttribute("xformOp:translate", pos);
        prim->setAttribute("xformOp:rotateQuaternion", rot);

        skelMesh._bodyPrims[idx] = prim;
    }

    fmt::print("Robot loaded: {} bodies, {} meshes\n",
               skelMesh._skeleton->numJoints(), mjcfData.meshInfos.size());

    return skelMesh;
}

void SkelMesh::applyPose() {
    auto globals = _state.computeGlobalTransforms();
    for (int i = 0; i < _skeleton->numJoints(); ++i) {
        if (!_bodyPrims[i])
            continue;
        glm::vec3 pos = toGlm(globals[i].translation) * _scale;
        glm::quat rot = toGlm(globals[i].rotation);
        _bodyPrims[i]->setAttribute("xformOp:translate", pos);
        _bodyPrims[i]->setAttribute("xformOp:rotateQuaternion", rot);
    }
}

void SkelMesh::setJointRotation(int idx, const Eigen::Quaternionf& q) {
    _state.setRotation(idx, q);
}

void SkelMesh::setRootTranslation(const Eigen::Vector3f& t) {
    _state.setRootTranslation(t);
}

void SkelMesh::resetToZeroPose() {
    _state = SkeletonState::zeroPose(_skeleton);
    applyPose();
}

} // namespace Animation
} // namespace KE

#include "robot.hpp"
#include "skeleton_math.hpp"
#include "utils/load_utils.hpp"
#include "scene/scene_backend.hpp"
#include "scene/native/prim.hpp"
#include "app/app.hpp"

#include <fmt/core.h>

namespace KE {
namespace Animation {

Robot Robot::fromMJCF(const std::string& mjcfPath, Scene::SceneBackend* scene,
                      Backend::Shader* shader, App* app,
                      const std::string& primBasePath, float scale) {
    Robot robot;
    robot._scale = scale;

    // 1. Load MJCF
    auto mjcfData = MJCFLoader::load(mjcfPath);
    robot._skeleton = std::make_shared<const SkeletonTree>(
        std::move(mjcfData.skeleton));
    robot._skeleton->print();

    // 2. Zero pose (rest pose)
    robot._state = SkeletonState::zeroPose(robot._skeleton);
    auto globalTransforms = robot._state.computeGlobalTransforms();
    robot._state.printGlobalPositions();

    // 3. Create prims per body with STL meshes
    robot._bodyPrims.resize(robot._skeleton->numJoints(), nullptr);

    for (const auto& meshInfo : mjcfData.meshInfos) {
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

        app->addShape(shader, prim);
        robot._bodyPrims[idx] = prim;
    }

    fmt::print("Robot loaded: {} bodies, {} meshes\n",
               robot._skeleton->numJoints(), mjcfData.meshInfos.size());

    return robot;
}

void Robot::applyPose() {
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

void Robot::setJointRotation(int idx, const Eigen::Quaternionf& q) {
    _state.setRotation(idx, q);
}

void Robot::setRootTranslation(const Eigen::Vector3f& t) {
    _state.setRootTranslation(t);
}

void Robot::resetToZeroPose() {
    _state = SkeletonState::zeroPose(_skeleton);
    applyPose();
}

} // namespace Animation
} // namespace KE

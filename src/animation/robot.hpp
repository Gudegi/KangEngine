#ifndef _ROBOT_HPP_
#define _ROBOT_HPP_

#include "mjcf_loader.hpp"
#include "skeleton_state.hpp"

#include <memory>
#include <string>
#include <vector>

namespace KE {

class App;

namespace Backend {
class Shader;
}

namespace Scene {
class SceneBackend;
class Prim;
} // namespace Scene

namespace Animation {

class Robot {
  public:
    // Factory: MJCF 로드 + Scene에 Prim 생성 + STL 메시 로드
    static Robot fromMJCF(const std::string& mjcfPath,
                          Scene::SceneBackend* scene,
                          Backend::Shader* shader, App* app,
                          const std::string& primBasePath = "/robot",
                          float scale = 1.0f);

    // FK 계산 후 모든 body Prim의 xformOp 업데이트
    void applyPose();

    // Pose 조작
    void setJointRotation(int idx, const Eigen::Quaternionf& q);
    void setRootTranslation(const Eigen::Vector3f& t);
    void resetToZeroPose();

    // Accessors
    const SkeletonTree& skeleton() const { return *_skeleton; }
    SkeletonState& state() { return _state; }
    const SkeletonState& state() const { return _state; }
    Scene::Prim* bodyPrim(int idx) const { return _bodyPrims[idx]; }
    int numBodies() const { return _skeleton->numJoints(); }

  private:
    std::shared_ptr<const SkeletonTree> _skeleton;
    SkeletonState _state;
    std::vector<Scene::Prim*> _bodyPrims; // per-joint, scene owns
    float _scale = 1.0f;
};

} // namespace Animation
} // namespace KE

#endif

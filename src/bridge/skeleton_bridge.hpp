///
/// SkeletonBridge — connects SkeletonFK pose computation to scene Prim visuals.
///

#ifndef _SKELETON_BRIDGE_HPP_
#define _SKELETON_BRIDGE_HPP_

#include "animation/skeleton_fk.hpp"
#include <string>
#include <vector>

namespace KE {

namespace Scene {
class SceneBackend;
class Prim;
} // namespace Scene

namespace Bridge {

class SkeletonBridge {
  public:
    SkeletonBridge() {}

    // Load MJCF: builds SkeletonFK + creates one scene Prim per body
    static SkeletonBridge fromMJCF(const std::string& mjcfPath,
                                   Scene::SceneBackend* scene,
                                   const std::string& primBasePath = "/robot",
                                   float scale = 1.0f,
                                   const std::string& order = "DFS");

    static SkeletonBridge fromData(const Animation::CharacterData& data,
                                   Scene::SceneBackend* scene,
                                   const std::string& primBasePath = "/robot",
                                   float scale = 1.0f);

    // Sync FK global transforms > update all body Prim xformOp attributes.
    void applyPose();

    // Pose manipulation (delegates to SkeletonFK)
    void setJointRotation(int idx, const Eigen::Quaternionf& q);
    void setRootTranslation(const Eigen::Vector3f& t);
    void resetToZeroPose();

    // Accessors
    Animation::SkeletonFK& fk() { return _fk; }
    const Animation::SkeletonFK& fk() const { return _fk; }
    const std::vector<Scene::Prim*>& bodyPrims() const { return _bodyPrims; }
    Scene::Prim* bodyPrim(int idx) const { return _bodyPrims[idx]; }
    int numBodies() const { return _fk.numBodies(); }

  private:
    Animation::SkeletonFK _fk;
    std::vector<Scene::Prim*> _bodyPrims; // non-owning, scene owns
};

} // namespace Bridge
} // namespace KE

#endif

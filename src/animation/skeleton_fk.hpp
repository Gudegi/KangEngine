///
/// Pure FK solver — skeleton topology + pose state, no visual/scene dependency.
///

#ifndef _SKELETON_FK_HPP_
#define _SKELETON_FK_HPP_

#include "mjcf_loader.hpp"
#include "skeleton_state.hpp"

#include <memory>
#include <string>

namespace KE {
namespace Animation {

class SkeletonFK {
  public:
    SkeletonFK() {}

    // Load skeleton from MJCF. Does not create any scene Prims.
    static SkeletonFK fromMJCF(const std::string& mjcfPath, float scale = 1.0f,
                               const std::string& order = "DFS");

    // Build from pre-parsed MJCFData (no extra file I/O).
    static SkeletonFK fromData(const MJCFData& data, float scale = 1.0f);

    void setJointRotation(int idx, const Eigen::Quaternionf& q);
    void setRootTranslation(const Eigen::Vector3f& t);
    void resetToZeroPose();

    const SkeletonTree& skeleton() const { return *_skeleton; }
    SkeletonState& state() { return _state; }
    const SkeletonState& state() const { return _state; }
    int numBodies() const { return _skeleton->numJoints(); }
    float scale() const { return _scale; }
    std::shared_ptr<const SkeletonTree> skeletonPtr() const {
        return _skeleton;
    }

  private:
    std::shared_ptr<const SkeletonTree> _skeleton;
    SkeletonState _state;
    float _scale = 1.0f;
};

} // namespace Animation
} // namespace KE

#endif

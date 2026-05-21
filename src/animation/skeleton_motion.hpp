#ifndef _SKELETON_MOTION_HPP_
#define _SKELETON_MOTION_HPP_

#include "skeleton_tree.hpp"
#include "skeleton_state.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

namespace KE {
namespace Animation {

class SkeletonMotion {
  public:
    SkeletonMotion() = default;
    SkeletonMotion(std::shared_ptr<const SkeletonTree> tree, float fps,
                   std::string motionName, std::vector<float> rootTranslations,
                   std::vector<float> localRotationsWxyz);

    int numFrames() const;
    float fps() const { return _fps; }
    float duration() const;
    const std::string& motionName() const { return _motionName; }

    const SkeletonTree& skeletonTree() const { return *_tree; }
    int numJoints() const { return _tree->numJoints(); }

    Eigen::Vector3f rootTranslation(int frameIndex) const;
    Eigen::Quaternionf localRotation(int frameIndex, int jointIndex) const;

    // Allocates a SkeletonState rotation vector. Prefer sample/apply-time paths
    // with reusable buffers for per-frame playback hot loops.
    SkeletonState frame(int frameIndex) const;
    SkeletonState sample(float time, bool loop = true) const;

    const std::vector<float>& rootTranslationsFlat() const {
        return _rootTranslations;
    }
    const std::vector<float>& localRotationsWxyzFlat() const {
        return _localRotationsWxyz;
    }

  private:
    float _fps = 30.0f;
    std::shared_ptr<const SkeletonTree> _tree;
    std::string _motionName;

    // Quaternion convention:
    // - Animation-core storage uses wxyz, matching Eigen::Quaternionf,
    //   glm::quat, MJCF quat attributes, and ke.quat constructor semantics.
    // - Python/numpy and PhysX flat boundary APIs use xyzw and must convert
    //   explicitly when data crosses that boundary.
    std::vector<float> _rootTranslations;   // [num_frames, 3]
    std::vector<float> _localRotationsWxyz; // [num_frames, num_joints, 4]
};

} // namespace Animation
} // namespace KE

#endif

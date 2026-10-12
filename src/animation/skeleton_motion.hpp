#ifndef _SKELETON_MOTION_HPP_
#define _SKELETON_MOTION_HPP_

#include "skeleton_tree.hpp"
#include "skeleton_state.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace KE {
namespace Animation {

class SkeletonMotion {
  public:
    // Copies share immutable frames and skeleton; sampled states are independent.
    SkeletonMotion() = default;
    SkeletonMotion(std::shared_ptr<const SkeletonTree> tree, float fps,
                   std::string motionName, std::vector<float> rootTranslations,
                   std::vector<float> localRotationsWxyz);

    int numFrames() const;
    float fps() const { return _fps; }
    float duration() const;
    const std::string& motionName() const { return _motionName; }

    const SkeletonTree& skeletonTree() const { return *_tree; }
    const std::shared_ptr<const SkeletonTree>& skeletonTreePtr() const {
        return _tree;
    }
    int numJoints() const { return _tree->numJoints(); }

    Eigen::Vector3f rootTranslation(int frameIndex) const;
    Eigen::Quaternionf localRotation(int frameIndex, int jointIndex) const;

    // Allocates a SkeletonState rotation vector. Prefer sample/apply-time paths
    // with reusable buffers for per-frame playback hot loops.
    SkeletonState frame(int frameIndex) const;
    SkeletonState sample(float time, bool loop = true) const;
    // Initializes empty output or reuses a compatible local state; otherwise throws.
    // Looping holds the last frame for 1/fps seconds, without blending to the first.
    void sampleInto(double time, SkeletonState& output, bool loop = true) const;

    // Checks finite positions and unit rotations once per shared frame buffer.
    void validateFrameData() const;

    std::vector<Transform> globalTransforms() const;
    std::vector<Eigen::Vector3f> globalPositions() const;
    std::vector<Eigen::Vector3f> globalLinearVelocities() const;
    std::vector<Eigen::Vector3f> globalAngularVelocities() const;
    std::vector<Eigen::Vector3f> globalLinearAccelerations() const;
    std::vector<Eigen::Vector3f> globalAngularAccelerations() const;
    std::vector<Eigen::Vector3f> rootLinearVelocities() const;
    std::vector<Eigen::Vector3f> rootLinearAccelerations() const;

    const std::vector<float>& rootTranslationsFlat() const {
        return data().rootTranslations;
    }
    const std::vector<float>& localRotationsWxyzFlat() const {
        return data().localRotationsWxyz;
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
    struct Data {
        Data() = default;
        Data(std::vector<float> translations, std::vector<float> rotations)
            : rootTranslations(std::move(translations)),
              localRotationsWxyz(std::move(rotations)) {}

        std::vector<float> rootTranslations;   // [num_frames, 3]
        std::vector<float> localRotationsWxyz; // [num_frames, num_joints, 4]
        mutable std::once_flag validation;
    };
    std::shared_ptr<const Data> _data;

    const Data& data() const {
        // Allocation-free fallback for default and moved-from motions.
        static const Data empty;
        return _data ? *_data : empty;
    }
};

} // namespace Animation
} // namespace KE

#endif

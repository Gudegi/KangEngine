#ifndef _SKELETON_STATE_HPP_
#define _SKELETON_STATE_HPP_

#include "skeleton_tree.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

namespace KE {
namespace Animation {

struct Transform {
    Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
};

class SkeletonState {
  public:
    // Factory methods
    static SkeletonState fromRotationAndRootTranslation(
        std::shared_ptr<const SkeletonTree> tree,
        const std::vector<Eigen::Quaternionf>& rotations,
        const Eigen::Vector3f& rootTranslation, bool isLocal = true);

    static SkeletonState zeroPose(std::shared_ptr<const SkeletonTree> tree);

    // Forward kinematics: local rotations -> global transforms
    std::vector<Transform> computeGlobalTransforms() const;

    // Global joint positions (convenience)
    std::vector<Eigen::Vector3f> computeGlobalPositions() const;

    // Accessors
    const SkeletonTree& skeletonTree() const { return *_tree; }
    int numJoints() const { return _tree->numJoints(); }
    bool isLocal() const { return _isLocal; }

    const Eigen::Quaternionf& rotation(int i) const { return _rotations[i]; }
    void setRotation(int i, const Eigen::Quaternionf& q) { _rotations[i] = q; }

    const Eigen::Vector3f& rootTranslation() const { return _rootTranslation; }
    void setRootTranslation(const Eigen::Vector3f& t) { _rootTranslation = t; }

    const std::vector<Eigen::Quaternionf>& rotations() const {
        return _rotations;
    }

    // Debug print
    void printGlobalPositions() const;

  private:
    std::shared_ptr<const SkeletonTree> _tree;
    std::vector<Eigen::Quaternionf> _rotations;
    Eigen::Vector3f _rootTranslation = Eigen::Vector3f::Zero();
    bool _isLocal = true;
};

} // namespace Animation
} // namespace KE

#endif

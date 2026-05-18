#include "skeleton_state.hpp"

#include <fmt/core.h>

namespace KE {
namespace Animation {

SkeletonState SkeletonState::fromRotationAndRootTranslation(
    std::shared_ptr<const SkeletonTree> tree,
    const std::vector<Eigen::Quaternionf>& rotations,
    const Eigen::Vector3f& rootTranslation, bool isLocal) {
    SkeletonState state;
    state._tree = std::move(tree);
    state._rotations = rotations;
    state._rootTranslation = rootTranslation;
    state._isLocal = isLocal;
    return state;
}

SkeletonState
SkeletonState::zeroPose(std::shared_ptr<const SkeletonTree> tree) {
    std::vector<Eigen::Quaternionf> rotations(tree->localRotations().begin(),
                                              tree->localRotations().end());
    Eigen::Vector3f rootPos = tree->localTranslation(0);
    return fromRotationAndRootTranslation(std::move(tree), rotations, rootPos,
                                          true);
}

std::vector<Transform> SkeletonState::computeGlobalTransforms() const {
    int n = _tree->numJoints();
    std::vector<Transform> global(n);

    for (int i = 0; i < n; ++i) {
        const Eigen::Quaternionf& localRot = _rotations[i];
        const Eigen::Vector3f& localTrans = _tree->localTranslation(i);
        int parent = _tree->parentIndex(i);

        if (parent == -1) {
            // Root node: use root translation + local offset
            global[i].rotation = localRot;
            // global[i].translation = _rootTranslation + localRot * localTrans;
            global[i].translation = _rootTranslation;
        } else {
            // Child: compose with parent's global transform
            // global_rot = parent_rot * local_rot
            // global_pos = parent_pos + parent_rot * local_trans
            global[i].rotation = global[parent].rotation * localRot;
            global[i].translation = global[parent].translation +
                                    global[parent].rotation * localTrans;
        }
    }

    return global;
}

std::vector<Eigen::Vector3f> SkeletonState::computeGlobalPositions() const {
    auto transforms = computeGlobalTransforms();
    std::vector<Eigen::Vector3f> positions(transforms.size());
    for (size_t i = 0; i < transforms.size(); ++i) {
        positions[i] = transforms[i].translation;
    }
    return positions;
}

void SkeletonState::printGlobalPositions() const {
    auto positions = computeGlobalPositions();
    fmt::print("Global joint positions ({} joints):\n",
               static_cast<int>(positions.size()));
    for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
        const auto& p = positions[i];
        fmt::print("  [{}] {}: ({:.4f}, {:.4f}, {:.4f})\n", i,
                   _tree->nodeName(i), p.x(), p.y(), p.z());
    }
}

} // namespace Animation
} // namespace KE

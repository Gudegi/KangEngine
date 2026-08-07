#pragma once

#include "skeleton_motion.hpp"
#include "skeleton_state.hpp"

#include <Eigen/Core>
#include <vector>

namespace KE::Animation {

struct IKEffector {
    /// Joint whose global-space point should reach the target.
    int joint = -1;
    /// Point expressed in the joint's local coordinates.
    Eigen::Vector3f offset = Eigen::Vector3f::Zero();
};

struct IKJointControl {
    /// Controlled joint. A joint may appear in at most one control group.
    int joint = -1;
    /// One to three local rotation axes, applied in the listed order.
    std::vector<Eigen::Vector3f> axes;
};

struct FullBodyIKConfig {
    /// Zero lets LBFGS++ iterate until its convergence criteria are met.
    int maxIterations = 0;
    int historySize = 6;
    int maxLineSearch = 40;
    double gradientTolerance = 1e-9;
    double relativeTolerance = 1e-9;
    /// Quadratic penalty that keeps the solution near the source pose.
    double poseDamping = 0.01;
    /// Maximum worker count used by the motion batch solver.
    int maxThreads = 16;
};

struct FullBodyIKResult {
    SkeletonState state;
    std::vector<float> bodyPositions; // [joint, xyz]
    std::vector<float> finalErrors;   // [effector]
    int iterations = 0;              // Negative on line-search failure
};

struct FullBodyIKBatchResult {
    SkeletonMotion motion;
    std::vector<float> bodyPositions; // [frame, joint, xyz]
    std::vector<float> finalErrors;   // [frame, effector]
    std::vector<int> iterations; // [frame], negative on line-search failure
};

FullBodyIKResult solveFullBodyIK(
    const SkeletonState& state,
    const std::vector<Eigen::Vector3f>& targetPositions,
    const std::vector<IKEffector>& effectors,
    const std::vector<IKJointControl>& controls,
    const FullBodyIKConfig& config = {});

FullBodyIKBatchResult solveFullBodyIKBatch(
    const SkeletonMotion& motion, const std::vector<float>& targetPositions,
    const std::vector<IKEffector>& effectors,
    const std::vector<IKJointControl>& controls,
    const FullBodyIKConfig& config = {});

} // namespace KE::Animation

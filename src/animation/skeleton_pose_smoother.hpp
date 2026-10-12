#pragma once

#include "skeleton_state.hpp"

namespace KE::Animation {

// State(Local joint rotations with root pose/linear velocity in world space.)
// Angular velocities: rad/s in parent frames (world for root), not
// rotation-vector rates.
struct SkeletonPoseSample {
    SkeletonState state;
    Eigen::Vector3f rootLinearVelocity = Eigen::Vector3f::Zero();
    std::vector<Eigen::Vector3f> localAngularVelocities;
};

// Decays transition offsets while following the supplied pose and velocities.
class SkeletonPoseSmoother {
  public:
    explicit SkeletonPoseSmoother(const SkeletonPoseSample& initial,
                                  float halfLife = 0.1f);

    void reset(const SkeletonPoseSample& target);
    // Preserves output pose/velocity; halfLife=0 snaps to target.
    void transition(const SkeletonPoseSample& target);
    // Target must already be advanced to t+dt. dt=0 validates without changing
    // state.
    void advance(const SkeletonPoseSample& target, double dt);

    const SkeletonPoseSample& sample() const { return _output; }
    float halfLife() const { return _halfLife; }
    void setHalfLife(float halfLife);

  private:
    void validateSample(const SkeletonPoseSample& sample) const;
    void copyTarget(const SkeletonPoseSample& target);
    void clearOffsets();
    void updateOutput();

    SkeletonPoseSample _target;
    SkeletonPoseSample _output;
    float _halfLife;
    Eigen::Vector3f _rootOffset = Eigen::Vector3f::Zero();
    Eigen::Vector3f _rootOffsetVelocity = Eigen::Vector3f::Zero();
    std::vector<Eigen::Vector3f> _rotationOffsets;
    std::vector<Eigen::Vector3f> _rotationOffsetVelocities;
};

} // namespace KE::Animation

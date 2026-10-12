#pragma once

#include "skeleton_motion.hpp"
#include "skeleton_pose_smoother.hpp"
#include <optional>

namespace KE {
namespace Animation {

// Clip playback and alignment with SkeletonPoseSmoother transitions.
// Players share immutable clips and own independent playback state.
class SkeletonMotionPlayer {
  public:
    // Starts at clip 0, time 0; uses source placement if heading is undefined.
    explicit SkeletonMotionPlayer(std::vector<SkeletonMotion> motions,
                                  int upAxis = 2, int forwardAxis = 0,
                                  float transitionHalfLife = 0.1f);

    // Time is seconds in [0, duration()]. Reset clears smoothing,
    // centers planar position and aligns heading to forwardAxis.
    void reset(int motionIndex = 0, double time = 0.0);
    // Aligns the new clip to the current planar position and heading.
    void transition(int motionIndex, double time);
    // Explicit source-to-world placement of root pose and velocities before smoothing.
    void resetPlaced(int motionIndex, double time,
                     const Eigen::Quaternionf& rotation,
                     const Eigen::Vector3f& translation);
    void transitionPlaced(int motionIndex, double time,
                          const Eigen::Quaternionf& rotation,
                          const Eigen::Vector3f& translation);
    // Finite dt >= 0; no looping. Smoothing continues after the clip ends.
    void advance(double dt);

    const SkeletonState& state() const { return _smoother->sample().state; }
    int motionIndex() const { return _motionIndex; }
    double time() const { return _time; }
    double duration() const;
    Eigen::Vector3f rootPosition() const { return state().rootTranslation(); }
    Eigen::Quaternionf headingRotation() const;

    // Zero clears/disables smoothing; positive changes preserve current offsets/velocities.
    float transitionHalfLife() const { return _smoother->halfLife(); }
    void setTransitionHalfLife(float halfLife);

    const Eigen::Vector3f& rootLinearVelocity() const {
        return _smoother->sample().rootLinearVelocity;
    }
    // Output angular velocities in parent frames; root uses world coordinates.
    const std::vector<Eigen::Vector3f>& localAngularVelocities() const {
        return _smoother->sample().localAngularVelocities;
    }

  private:
    double sampleSource(int motionIndex, double time);
    void placeTarget(int motionIndex, double time,
                     const Eigen::Quaternionf& rotation,
                     const Eigen::Vector3f& translation);
    void updateTarget();
    Eigen::Quaternionf projectHeading(const Eigen::Quaternionf& rotation) const;
    Eigen::Quaternionf sourceHeading(int motionIndex, double time) const;

    std::vector<SkeletonMotion> _motions;
    std::vector<SkeletonState> _samples;
    SkeletonPoseSample _target;
    std::optional<SkeletonPoseSmoother> _smoother;
    int _upAxis;
    int _forwardAxis;
    // Enables heading fallback when the smoothed forward axis is vertical.
    bool _walkingAlignment = false;
    int _motionIndex = 0;
    double _time = 0.0;
    Eigen::Quaternionf _alignmentRotation = Eigen::Quaternionf::Identity();
    Eigen::Vector3f _alignmentTranslation = Eigen::Vector3f::Zero();
};

} // namespace Animation
} // namespace KE

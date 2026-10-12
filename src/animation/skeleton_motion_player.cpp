#include "skeleton_motion_player.hpp"
#include "quaternion_math.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace KE {
namespace Animation {
namespace {

void validateHalfLife(float value) {
    if (!std::isfinite(value) || value < 0.0f)
        throw std::invalid_argument(
            "SkeletonMotionPlayer half-life must be finite and nonnegative");
}

void validateUnitQuaternion(const Eigen::Quaternionf& q) {
    if (!q.coeffs().allFinite() || std::abs(q.norm() - 1.0f) > 1e-3f)
        throw std::invalid_argument(
            "SkeletonMotionPlayer requires finite unit quaternions");
}

void validateTree(const SkeletonTree& tree) {
    const int joints = tree.numJoints();
    if (joints <= 0 ||
        tree.parentIndices().size() != static_cast<size_t>(joints) ||
        tree.localTranslations().size() != static_cast<size_t>(joints) ||
        tree.localRotations().size() != static_cast<size_t>(joints))
        throw std::invalid_argument(
            "SkeletonMotionPlayer requires a nonempty complete skeleton");
    std::unordered_set<std::string> names;
    for (int joint = 0; joint < joints; ++joint) {
        const int parent = tree.parentIndex(joint);
        if ((joint == 0 && parent != -1) ||
            (joint > 0 && (parent < 0 || parent >= joint)))
            throw std::invalid_argument(
                "SkeletonMotionPlayer requires one root first and parents "
                "before children");
        if (!names.insert(tree.nodeName(joint)).second)
            throw std::invalid_argument(
                "SkeletonMotionPlayer requires unique joint names");
        if (!tree.localTranslation(joint).allFinite())
            throw std::invalid_argument(
                "SkeletonMotionPlayer requires finite rest translations");
        validateUnitQuaternion(tree.localRotation(joint));
    }
}

void validateSameSkeleton(const SkeletonTree& tree,
                          const SkeletonTree& reference) {
    if (tree.nodeNames() != reference.nodeNames() ||
        tree.parentIndices() != reference.parentIndices())
        throw std::invalid_argument(
            "SkeletonMotionPlayer clips must share joint names and topology");
    for (int joint = 0; joint < tree.numJoints(); ++joint) {
        if ((tree.localTranslation(joint) - reference.localTranslation(joint))
                    .norm() > 1e-5f ||
            quatToRotationVector(
                tree.localRotation(joint).normalized() *
                reference.localRotation(joint).normalized().conjugate())
                    .norm() > 1e-5f)
            throw std::invalid_argument(
                "SkeletonMotionPlayer clips must share rest transforms");
    }
}

double clipDuration(const SkeletonMotion& motion) {
    return static_cast<double>(motion.numFrames() - 1) / motion.fps();
}

} // namespace

SkeletonMotionPlayer::SkeletonMotionPlayer(std::vector<SkeletonMotion> motions,
                                           int upAxis, int forwardAxis,
                                           float transitionHalfLife)
    : _motions(std::move(motions)), _upAxis(upAxis), _forwardAxis(forwardAxis) {
    validateHalfLife(transitionHalfLife);
    if (upAxis < 0 || upAxis > 2 || forwardAxis < 0 || forwardAxis > 2 ||
        upAxis == forwardAxis)
        throw std::invalid_argument("SkeletonMotionPlayer up and forward axes "
                                    "must be distinct indices in [0, 2]");
    if (_motions.empty())
        throw std::invalid_argument(
            "SkeletonMotionPlayer requires at least one motion");
    _samples.reserve(_motions.size());
    std::unordered_set<const SkeletonTree*> validatedTrees;
    for (const SkeletonMotion& motion : _motions) {
        if (!motion.skeletonTreePtr() || motion.numFrames() <= 0 ||
            !std::isfinite(motion.fps()) || motion.fps() <= 0.0f)
            throw std::invalid_argument(
                "SkeletonMotionPlayer requires nonempty motions with positive "
                "finite fps");
        if (validatedTrees.insert(motion.skeletonTreePtr().get()).second) {
            validateTree(motion.skeletonTree());
            if (motion.skeletonTreePtr() != _motions.front().skeletonTreePtr())
                validateSameSkeleton(motion.skeletonTree(),
                                     _motions.front().skeletonTree());
        }
        motion.validateFrameData();
        _samples.push_back(SkeletonState::zeroPose(motion.skeletonTreePtr()));
    }
    const auto& tree = _motions.front().skeletonTreePtr();
    _target.state = SkeletonState::zeroPose(tree);
    _target.localAngularVelocities.resize(
        static_cast<size_t>(tree->numJoints()), Eigen::Vector3f::Zero());
    _smoother.emplace(_target, transitionHalfLife);
    // Explicit placement also supports clips without a planar heading.
    Eigen::Vector3f forward =
        _motions.front().localRotation(0, 0) * Eigen::Vector3f::Unit(_forwardAxis);
    forward[_upAxis] = 0.0f;
    if (forward.squaredNorm() >= 1e-12f)
        reset();
    else
        resetPlaced(0, 0.0, Eigen::Quaternionf::Identity(),
                    Eigen::Vector3f::Zero());
}

double SkeletonMotionPlayer::duration() const {
    return clipDuration(_motions[static_cast<size_t>(_motionIndex)]);
}

double SkeletonMotionPlayer::sampleSource(int motionIndex, double time) {
    if (motionIndex < 0 || static_cast<size_t>(motionIndex) >= _motions.size())
        throw std::out_of_range(
            "SkeletonMotionPlayer motion index out of range");
    const SkeletonMotion& motion = _motions[static_cast<size_t>(motionIndex)];
    // Accept float-rounded duration(), then clamp to the exact endpoint.
    const double acceptedEnd =
        std::max(clipDuration(motion), static_cast<double>(motion.duration()));
    if (!std::isfinite(time) || time < 0.0 || time > acceptedEnd)
        throw std::invalid_argument("SkeletonMotionPlayer clip time must be "
                                    "finite and within its duration");
    time = std::min(time, clipDuration(motion));
    motion.sampleInto(time, _samples[static_cast<size_t>(motionIndex)], false);
    return time;
}

Eigen::Quaternionf
SkeletonMotionPlayer::projectHeading(const Eigen::Quaternionf& rotation) const {
    const Eigen::Vector3f up = Eigen::Vector3f::Unit(_upAxis);
    const Eigen::Vector3f forward = Eigen::Vector3f::Unit(_forwardAxis);
    Eigen::Vector3f direction = rotation * forward;
    direction[_upAxis] = 0.0f;
    if (!direction.allFinite() || direction.squaredNorm() < 1e-12f)
        throw std::invalid_argument("SkeletonMotionPlayer root forward has a "
                                    "degenerate planar projection");
    direction.normalize();
    const float angle =
        std::atan2(up.dot(forward.cross(direction)), forward.dot(direction));
    return Eigen::Quaternionf(Eigen::AngleAxisf(angle, up));
}

Eigen::Quaternionf SkeletonMotionPlayer::sourceHeading(int motionIndex,
                                                       double time) const {
    const auto& motion = _motions[static_cast<size_t>(motionIndex)];
    const double frameFloat = time * motion.fps();
    const int frame = std::clamp(static_cast<int>(std::floor(frameFloat)), 0,
                                 motion.numFrames() - 1);
    const float alpha = static_cast<float>(frameFloat - frame);
    if (alpha == 0.0f || frame == motion.numFrames() - 1)
        return projectHeading(motion.localRotation(frame, 0));
    // Interpolate before projection to preserve heading for tilted roots.
    const Eigen::Quaternionf before = motion.localRotation(frame, 0);
    const Eigen::Quaternionf after =
        alignQuaternion(before, motion.localRotation(frame + 1, 0));
    const Eigen::Quaternionf rotation = before.slerp(alpha, after).normalized();
    Eigen::Vector3f direction = rotation * Eigen::Vector3f::Unit(_forwardAxis);
    direction[_upAxis] = 0.0f;
    if (direction.squaredNorm() < 1e-12f) {
        return projectHeading(motion.localRotation(frame, 0))
            .slerp(alpha, projectHeading(motion.localRotation(frame + 1, 0)))
            .normalized();
    }
    return projectHeading(rotation);
}

Eigen::Quaternionf SkeletonMotionPlayer::headingRotation() const {
    Eigen::Vector3f forward =
        state().rotation(0) * Eigen::Vector3f::Unit(_forwardAxis);
    forward[_upAxis] = 0.0f;
    if (forward.squaredNorm() >= 1e-12f)
        return projectHeading(state().rotation(0));
    if (!_walkingAlignment)
        throw std::invalid_argument(
            "heading is undefined for a vertical projected forward axis; use "
            "the full root rotation");
    // Fall back to aligned source heading while smoothing makes it undefined.
    const auto& motion = _motions[static_cast<size_t>(_motionIndex)];
    const double frameFloat = _time * motion.fps();
    const int before = std::clamp(static_cast<int>(std::floor(frameFloat)), 0,
                                  motion.numFrames() - 1);
    const int after = std::min(before + 1, motion.numFrames() - 1);
    const float alpha = static_cast<float>(frameFloat - before);
    return (_alignmentRotation *
            projectHeading(motion.localRotation(before, 0)).slerp(
                alpha, projectHeading(motion.localRotation(after, 0))))
        .normalized();
}

void SkeletonMotionPlayer::reset(int motionIndex, double time) {
    time = sampleSource(motionIndex, time);
    const auto& sample = _samples[static_cast<size_t>(motionIndex)];
    const Eigen::Quaternionf alignment =
        sourceHeading(motionIndex, time).conjugate();
    Eigen::Vector3f translation = -(alignment * sample.rootTranslation());
    translation[_upAxis] = 0.0f;
    placeTarget(motionIndex, time, alignment, translation);
    _smoother->reset(_target);
    _walkingAlignment = true;
}

void SkeletonMotionPlayer::resetPlaced(int motionIndex, double time,
                                       const Eigen::Quaternionf& rotation,
                                       const Eigen::Vector3f& translation) {
    time = sampleSource(motionIndex, time);
    placeTarget(motionIndex, time, rotation, translation);
    _smoother->reset(_target);
    _walkingAlignment = false;
}

void SkeletonMotionPlayer::transition(int motionIndex, double time) {
    time = sampleSource(motionIndex, time);
    const auto& sample = _samples[static_cast<size_t>(motionIndex)];
    const Eigen::Quaternionf alignment =
        (headingRotation() * sourceHeading(motionIndex, time).conjugate())
            .normalized();
    Eigen::Vector3f translation =
        state().rootTranslation() - alignment * sample.rootTranslation();
    translation[_upAxis] = 0.0f;
    placeTarget(motionIndex, time, alignment, translation);
    _smoother->transition(_target);
    _walkingAlignment = true;
}

void SkeletonMotionPlayer::transitionPlaced(
    int motionIndex, double time, const Eigen::Quaternionf& rotation,
    const Eigen::Vector3f& translation) {
    time = sampleSource(motionIndex, time);
    placeTarget(motionIndex, time, rotation, translation);
    _smoother->transition(_target);
    _walkingAlignment = false;
}

void SkeletonMotionPlayer::placeTarget(
    int motionIndex, double time, const Eigen::Quaternionf& rotation,
    const Eigen::Vector3f& translation) {
    validateUnitQuaternion(rotation);
    if (!translation.allFinite())
        throw std::invalid_argument("placement translation must be finite");
    _alignmentRotation = rotation.normalized();
    _alignmentTranslation = translation;
    _motionIndex = motionIndex;
    _time = time;
    updateTarget();
}

void SkeletonMotionPlayer::updateTarget() {
    const size_t index = static_cast<size_t>(_motionIndex);
    const SkeletonMotion& motion = _motions[index];
    const SkeletonState& sample = _samples[index];
    _target.state.setRootTranslation(
        _alignmentRotation * sample.rootTranslation() + _alignmentTranslation);
    for (int joint = 0; joint < motion.numJoints(); ++joint) {
        _target.state.setRotation(
            joint,
            joint == 0
                ? (_alignmentRotation * sample.rotation(joint)).normalized()
                : sample.rotation(joint));
    }
    if (_time >= duration() || motion.numFrames() == 1) {
        _target.rootLinearVelocity.setZero();
        for (auto& velocity : _target.localAngularVelocities)
            velocity.setZero();
        return;
    }
    // Right derivative of interpolation; source frame velocity jumps remain.
    const int before =
        std::clamp(static_cast<int>(std::floor(_time * motion.fps())), 0,
                   motion.numFrames() - 2);
    const int after = before + 1;
    _target.rootLinearVelocity =
        _alignmentRotation *
        ((motion.rootTranslation(after) - motion.rootTranslation(before)) *
         motion.fps());
    for (int joint = 0; joint < motion.numJoints(); ++joint) {
        const Eigen::Quaternionf q0 = motion.localRotation(before, joint);
        const Eigen::Quaternionf q1 =
            alignQuaternion(q0, motion.localRotation(after, joint));
        Eigen::Quaternionf difference = q1 * q0.conjugate();
        // Match slerp's scalar rounding at the pi boundary.
        difference.w() = q0.dot(q1);
        Eigen::Vector3f velocity =
            quatToRotationVector(difference) * motion.fps();
        if (joint == 0)
            velocity = _alignmentRotation * velocity;
        _target.localAngularVelocities[static_cast<size_t>(joint)] = velocity;
    }
}

void SkeletonMotionPlayer::advance(double dt) {
    if (!std::isfinite(dt) || dt < 0.0)
        throw std::invalid_argument(
            "SkeletonMotionPlayer dt must be finite and nonnegative");
    if (dt == 0.0)
        return;
    _time += std::min(dt, duration() - _time);
    const size_t index = static_cast<size_t>(_motionIndex);
    _motions[index].sampleInto(_time, _samples[index], false);
    updateTarget();
    _smoother->advance(_target, dt);
}

void SkeletonMotionPlayer::setTransitionHalfLife(float halfLife) {
    _smoother->setHalfLife(halfLife);
}

} // namespace Animation
} // namespace KE

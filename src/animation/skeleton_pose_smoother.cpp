#include "skeleton_pose_smoother.hpp"
#include "quaternion_math.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace KE::Animation {
namespace {
constexpr double kLogTwo = 0.69314718055994530942;

void decayOffset(Eigen::Vector3f& position, Eigen::Vector3f& velocity,
                 double lambda, double dt) {
    const double exponent = lambda * dt;
    // Avoid 0*inf when decay underflows.
    if (exponent > 700.0) {
        position.setZero();
        velocity.setZero();
        return;
    }
    const Eigen::Vector3d x = position.cast<double>();
    const Eigen::Vector3d v = velocity.cast<double>();
    const Eigen::Vector3d j = v + lambda * x;
    const double decay = std::exp(-exponent);
    position = ((x + j * dt) * decay).cast<float>();
    velocity = ((v - j * exponent) * decay).cast<float>();
}

void validateHalfLife(float value) {
    if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("half_life must be finite and nonnegative");
}

void validateTree(const SkeletonState& state) {
    if (!state.skeletonTreePtr())
        throw std::invalid_argument("pose sample requires a skeleton");
    const auto& tree = state.skeletonTree();
    const auto joints = static_cast<size_t>(tree.numJoints());
    if (!joints || tree.parentIndices().size() != joints ||
        tree.localTranslations().size() != joints ||
        tree.localRotations().size() != joints)
        throw std::invalid_argument(
            "pose sample requires a complete nonempty skeleton");
    std::unordered_set<std::string> names;
    for (size_t i = 0; i < joints; ++i) {
        const auto joint = static_cast<int>(i);
        const int parent = tree.parentIndex(joint);
        if ((i == 0 && parent != -1) ||
            (i > 0 && (parent < 0 || parent >= joint)) ||
            tree.nodeName(joint).empty() ||
            !names.insert(tree.nodeName(joint)).second)
            throw std::invalid_argument("pose sample requires unique names and "
                                        "one root in parent-first order");
        const auto& q = tree.localRotation(joint);
        if (!tree.localTranslation(joint).allFinite() ||
            !q.coeffs().allFinite() || std::abs(q.norm() - 1.0f) > 1e-3f)
            throw std::invalid_argument("pose sample requires finite rest "
                                        "transforms and unit quaternions");
    }
}
} // namespace

SkeletonPoseSmoother::SkeletonPoseSmoother(const SkeletonPoseSample& initial,
                                           float halfLife)
    : _target(initial), _output(initial), _halfLife(halfLife) {
    validateHalfLife(halfLife);
    validateTree(initial.state);
    validateSample(initial);
    const auto joints = initial.localAngularVelocities.size();
    _rotationOffsets.resize(joints, Eigen::Vector3f::Zero());
    _rotationOffsetVelocities.resize(joints, Eigen::Vector3f::Zero());
    copyTarget(initial);
    updateOutput();
}

void SkeletonPoseSmoother::validateSample(
    const SkeletonPoseSample& sample) const {
    const auto& state = sample.state;
    if (!state.skeletonTreePtr() || !state.isLocal() ||
        state.rotations().size() !=
            static_cast<size_t>(_output.state.numJoints()) ||
        sample.localAngularVelocities.size() != state.rotations().size() ||
        !state.rootTranslation().allFinite() ||
        !sample.rootLinearVelocity.allFinite())
        throw std::invalid_argument("pose sample requires compatible local "
                                    "rotations and finite root pose/velocity");
    if (state.skeletonTreePtr() != _output.state.skeletonTreePtr()) {
        const auto& tree = state.skeletonTree();
        const auto& reference = _output.state.skeletonTree();
        if (tree.nodeNames() != reference.nodeNames() ||
            tree.parentIndices() != reference.parentIndices() ||
            tree.localTranslations().size() !=
                reference.localTranslations().size() ||
            tree.localRotations().size() != reference.localRotations().size())
            throw std::invalid_argument(
                "pose sample skeleton does not match transition skeleton");
        for (int joint = 0; joint < tree.numJoints(); ++joint) {
            const auto& q = tree.localRotation(joint);
            if (!tree.localTranslation(joint).allFinite() ||
                !q.coeffs().allFinite() || std::abs(q.norm() - 1.0f) > 1e-3f ||
                (tree.localTranslation(joint) -
                 reference.localTranslation(joint))
                        .norm() > 1e-5f ||
                quatToRotationVector(
                    q.normalized() *
                    reference.localRotation(joint).normalized().conjugate())
                        .norm() > 1e-5f)
                throw std::invalid_argument("pose sample rest transforms do "
                                            "not match transition skeleton");
        }
    }
    for (size_t joint = 0; joint < state.rotations().size(); ++joint) {
        const auto& q = state.rotation(static_cast<int>(joint));
        if (!q.coeffs().allFinite() || std::abs(q.norm() - 1.0f) > 1e-3f ||
            !sample.localAngularVelocities[joint].allFinite())
            throw std::invalid_argument("pose sample requires unit rotations "
                                        "and finite angular velocities");
    }
}

void SkeletonPoseSmoother::copyTarget(const SkeletonPoseSample& target) {
    _target.state.setRootTranslation(target.state.rootTranslation());
    _target.rootLinearVelocity = target.rootLinearVelocity;
    for (size_t joint = 0; joint < _rotationOffsets.size(); ++joint) {
        _target.state.setRotation(
            static_cast<int>(joint),
            target.state.rotation(static_cast<int>(joint)).normalized());
        _target.localAngularVelocities[joint] =
            target.localAngularVelocities[joint];
    }
}

void SkeletonPoseSmoother::clearOffsets() {
    _rootOffset.setZero();
    _rootOffsetVelocity.setZero();
    for (size_t joint = 0; joint < _rotationOffsets.size(); ++joint) {
        _rotationOffsets[joint].setZero();
        _rotationOffsetVelocities[joint].setZero();
    }
}

void SkeletonPoseSmoother::reset(const SkeletonPoseSample& target) {
    validateSample(target);
    copyTarget(target);
    clearOffsets();
    updateOutput();
}

void SkeletonPoseSmoother::transition(const SkeletonPoseSample& target) {
    validateSample(target);
    copyTarget(target);
    if (_halfLife == 0) {
        clearOffsets();
        updateOutput();
        return;
    }
    _rootOffset =
        _output.state.rootTranslation() - _target.state.rootTranslation();
    _rootOffsetVelocity =
        _output.rootLinearVelocity - _target.rootLinearVelocity;
    for (size_t joint = 0; joint < _rotationOffsets.size(); ++joint) {
        const auto difference =
            (_output.state.rotation(static_cast<int>(joint)) *
             _target.state.rotation(static_cast<int>(joint)).conjugate())
                .normalized();
        _rotationOffsets[joint] = quatToRotationVector(difference);
        const Eigen::Vector3f relativeVelocity =
            _output.localAngularVelocities[joint] -
            difference * _target.localAngularVelocities[joint];
        _rotationOffsetVelocities[joint] =
            inverseLeftJacobian(_rotationOffsets[joint], relativeVelocity);
    }
    // Preserve output until advance(), including interrupted transitions.
}

void SkeletonPoseSmoother::updateOutput() {
    _output.state.setRootTranslation(_target.state.rootTranslation() +
                                     _rootOffset);
    _output.rootLinearVelocity =
        _target.rootLinearVelocity + _rootOffsetVelocity;
    for (size_t joint = 0; joint < _rotationOffsets.size(); ++joint) {
        const auto offset = quatFromRotationVector(_rotationOffsets[joint]);
        _output.state.setRotation(
            static_cast<int>(joint),
            (offset * _target.state.rotation(static_cast<int>(joint)))
                .normalized());
        _output.localAngularVelocities[joint] =
            leftJacobian(_rotationOffsets[joint],
                         _rotationOffsetVelocities[joint]) +
            offset * _target.localAngularVelocities[joint];
    }
}

void SkeletonPoseSmoother::advance(const SkeletonPoseSample& target,
                                   double dt) {
    if (!std::isfinite(dt) || dt < 0)
        throw std::invalid_argument(
            "dt must be finite and nonnegative");
    validateSample(target);
    if (dt == 0)
        return;
    copyTarget(target);
    if (_halfLife > 0) {
        const double lambda = 2.0 * kLogTwo / _halfLife;
        decayOffset(_rootOffset, _rootOffsetVelocity, lambda, dt);
        for (size_t joint = 0; joint < _rotationOffsets.size(); ++joint)
            decayOffset(_rotationOffsets[joint],
                        _rotationOffsetVelocities[joint], lambda, dt);
    }
    updateOutput();
}

void SkeletonPoseSmoother::setHalfLife(float halfLife) {
    validateHalfLife(halfLife);
    _halfLife = halfLife;
    if (halfLife == 0) {
        clearOffsets();
        updateOutput();
    }
}
} // namespace KE::Animation

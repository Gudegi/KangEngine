#include "articulation_motion_mapper.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace KE::Animation {
namespace {

constexpr float kEpsilon = 1e-8f;
constexpr float kStep = 1e-4f;

Eigen::Quaternionf normalized(Eigen::Quaternionf q) {
    if (q.squaredNorm() < kEpsilon)
        return Eigen::Quaternionf::Identity();
    q.normalize();
    if (q.w() < 0.0f)
        q.coeffs() *= -1.0f;
    return q;
}

Eigen::Vector3f rotationVector(Eigen::Quaternionf q) {
    q = normalized(q);
    const float norm = q.vec().norm();
    if (norm < kEpsilon)
        return Eigen::Vector3f::Zero();
    return q.vec() * (2.0f * std::atan2(norm, q.w()) / norm);
}

using BlockList = std::vector<const ArticulationCoordinateBlock*>;

Eigen::Quaternionf compose(const BlockList& blocks,
                           const Eigen::VectorXf& angles) {
    Eigen::Quaternionf result = Eigen::Quaternionf::Identity();
    for (int i = 0; i < angles.size(); ++i) {
        Eigen::Vector3f axis = blocks[static_cast<size_t>(i)]->axis;
        if (axis.squaredNorm() < kEpsilon)
            throw std::runtime_error(
                "Revolute coordinate axis must be non-zero");
        result *=
            Eigen::Quaternionf(Eigen::AngleAxisf(angles[i], axis.normalized()));
    }
    return normalized(result);
}

Eigen::VectorXf decompose(const BlockList& blocks,
                          const Eigen::Quaternionf& target,
                          bool clampToLimits) {
    const int count = static_cast<int>(blocks.size());
    Eigen::VectorXf angles = Eigen::VectorXf::Zero(count);
    if (count == 1) {
        const Eigen::Vector3f axis = blocks[0]->axis.normalized();
        const Eigen::Quaternionf q = normalized(target);
        angles[0] = 2.0f * std::atan2(q.vec().dot(axis), q.w());
    }
    if (clampToLimits)
        for (int i = 0; i < count; ++i) {
            const auto* block = blocks[static_cast<size_t>(i)];
            angles[i] =
                std::clamp(angles[i], block->lowerLimit, block->upperLimit);
        }

    for (int iteration = 0; iteration < 24; ++iteration) {
        const Eigen::Vector3f error =
            rotationVector(compose(blocks, angles).conjugate() * target);
        if (error.norm() < 1e-6f)
            break;
        Eigen::MatrixXf jacobian(3, count);
        for (int column = 0; column < count; ++column) {
            Eigen::VectorXf perturbed = angles;
            perturbed[column] += kStep;
            const Eigen::Vector3f next =
                rotationVector(compose(blocks, perturbed).conjugate() * target);
            jacobian.col(column) = (next - error) / kStep;
        }
        const Eigen::MatrixXf normal =
            jacobian.transpose() * jacobian +
            1e-6f * Eigen::MatrixXf::Identity(count, count);
        const Eigen::VectorXf delta =
            normal.ldlt().solve(-jacobian.transpose() * error);
        if (!delta.allFinite())
            break;
        angles += delta;
        if (clampToLimits)
            for (int i = 0; i < count; ++i) {
                const auto* block = blocks[static_cast<size_t>(i)];
                angles[i] =
                    std::clamp(angles[i], block->lowerLimit, block->upperLimit);
            }
        if (delta.norm() < 1e-7f)
            break;
    }
    return angles;
}

} // namespace

ArticulationMotionMapper::ArticulationMotionMapper(
    ArticulationCoordinateLayout layout)
    : _layout(std::move(layout)) {}

ArticulationMappingResult ArticulationMotionMapper::toArticulationCoordinates(
    const SkeletonState& state, bool clampToLimits) const {
    if (!state.isLocal())
        throw std::invalid_argument(
            "ArticulationMotionMapper requires local SkeletonState rotations");

    ArticulationMappingResult result;
    result.q.assign(static_cast<size_t>(_layout.nq()), 0.0f);
    result.residualAngles.assign(static_cast<size_t>(state.numJoints()), 0.0f);
    std::unordered_map<int, BlockList> revoluteGroups;

    for (const auto& block : _layout.blocks()) {
        if (block.bodyIndex < 0 || block.bodyIndex >= state.numJoints())
            throw std::invalid_argument(
                "Coordinate block body index is outside SkeletonState");
        if (state.skeletonTree().nodeName(block.bodyIndex) != block.bodyName)
            throw std::invalid_argument(
                "SkeletonState does not match coordinate layout body names");

        const Eigen::Quaternionf current =
            normalized(state.rotation(block.bodyIndex));
        if (block.type == ArticulationCoordinateType::Free) {
            const Eigen::Vector3f root = state.rootTranslation();
            result.q[block.qOffset] = root.x();
            result.q[block.qOffset + 1] = root.y();
            result.q[block.qOffset + 2] = root.z();
            result.q[block.qOffset + 3] = current.w();
            result.q[block.qOffset + 4] = current.x();
            result.q[block.qOffset + 5] = current.y();
            result.q[block.qOffset + 6] = current.z();
        } else if (block.type == ArticulationCoordinateType::Spherical) {
            const Eigen::Quaternionf relative =
                normalized(block.referenceRotation.conjugate() * current);
            result.q[block.qOffset] = relative.w();
            result.q[block.qOffset + 1] = relative.x();
            result.q[block.qOffset + 2] = relative.y();
            result.q[block.qOffset + 3] = relative.z();
        } else if (block.type == ArticulationCoordinateType::Revolute) {
            revoluteGroups[block.bodyIndex].push_back(&block);
        }
    }

    for (const auto& [bodyIndex, blocks] : revoluteGroups) {
        const Eigen::Quaternionf relative =
            normalized(blocks[0]->referenceRotation.conjugate() *
                       state.rotation(bodyIndex));
        const Eigen::VectorXf angles =
            decompose(blocks, relative, clampToLimits);
        for (int i = 0; i < angles.size(); ++i)
            result.q[blocks[static_cast<size_t>(i)]->qOffset] = angles[i];
        const Eigen::Quaternionf residual =
            compose(blocks, angles).conjugate() * relative;
        result.residualAngles[static_cast<size_t>(bodyIndex)] =
            rotationVector(residual).norm();
    }
    return result;
}

SkeletonState
ArticulationMotionMapper::toSkeletonState(const std::vector<float>& q) const {
    if (static_cast<int>(q.size()) != _layout.nq())
        throw std::invalid_argument("q size does not match coordinate layout");
    const auto& tree = _layout.skeletonTreePtr();
    if (!tree)
        throw std::runtime_error("Coordinate layout has no SkeletonTree");

    std::vector<Eigen::Quaternionf> rotations(tree->localRotations().begin(),
                                              tree->localRotations().end());
    Eigen::Vector3f rootTranslation = tree->localTranslation(0);
    std::unordered_map<int, BlockList> revoluteGroups;

    for (const auto& block : _layout.blocks()) {
        if (block.type == ArticulationCoordinateType::Free) {
            rootTranslation = Eigen::Vector3f(
                q[block.qOffset], q[block.qOffset + 1], q[block.qOffset + 2]);
            rotations[static_cast<size_t>(block.bodyIndex)] = normalized(
                Eigen::Quaternionf(q[block.qOffset + 3], q[block.qOffset + 4],
                                   q[block.qOffset + 5], q[block.qOffset + 6]));
        } else if (block.type == ArticulationCoordinateType::Spherical) {
            const Eigen::Quaternionf relative = normalized(
                Eigen::Quaternionf(q[block.qOffset], q[block.qOffset + 1],
                                   q[block.qOffset + 2], q[block.qOffset + 3]));
            rotations[static_cast<size_t>(block.bodyIndex)] =
                normalized(block.referenceRotation * relative);
        } else if (block.type == ArticulationCoordinateType::Revolute) {
            revoluteGroups[block.bodyIndex].push_back(&block);
        } else if (block.type == ArticulationCoordinateType::Prismatic &&
                   std::abs(q[block.qOffset]) > kEpsilon) {
            throw std::runtime_error(
                "SkeletonState cannot represent prismatic joint translation");
        }
    }

    for (const auto& [bodyIndex, blocks] : revoluteGroups) {
        Eigen::VectorXf angles(static_cast<int>(blocks.size()));
        for (int i = 0; i < angles.size(); ++i)
            angles[i] = q[blocks[static_cast<size_t>(i)]->qOffset];
        rotations[static_cast<size_t>(bodyIndex)] =
            normalized(blocks[0]->referenceRotation * compose(blocks, angles));
    }

    return SkeletonState::fromRotationAndRootTranslation(
        tree, rotations, rootTranslation, true);
}

SkeletonMotion ArticulationMotionMapper::toSkeletonMotion(
    const ArticulationMotion& motion) const {
    if (motion.layout().modelSignature() != _layout.modelSignature())
        throw std::invalid_argument(
            "ArticulationMotion layout does not match mapper layout");
    const auto& tree = _layout.skeletonTreePtr();
    if (!tree)
        throw std::runtime_error("Coordinate layout has no SkeletonTree");

    const int frames = motion.numFrames();
    const int joints = tree->numJoints();
    std::vector<float> roots(static_cast<size_t>(frames) * 3);
    std::vector<float> rotations(static_cast<size_t>(frames) * joints * 4);
    std::vector<float> frameQ(static_cast<size_t>(_layout.nq()));
    const std::vector<float>& sourceQ = motion.qFlat();
    for (int frame = 0; frame < frames; ++frame) {
        const auto begin =
            sourceQ.begin() + static_cast<size_t>(frame) * _layout.nq();
        std::copy(begin, begin + _layout.nq(), frameQ.begin());
        const SkeletonState state = toSkeletonState(frameQ);
        const Eigen::Vector3f root = state.rootTranslation();
        const size_t rootOffset = static_cast<size_t>(frame) * 3;
        roots[rootOffset] = root.x();
        roots[rootOffset + 1] = root.y();
        roots[rootOffset + 2] = root.z();
        for (int joint = 0; joint < joints; ++joint) {
            const Eigen::Quaternionf rotation = state.rotation(joint);
            const size_t offset =
                (static_cast<size_t>(frame) * joints + joint) * 4;
            rotations[offset] = rotation.w();
            rotations[offset + 1] = rotation.x();
            rotations[offset + 2] = rotation.y();
            rotations[offset + 3] = rotation.z();
        }
    }
    return SkeletonMotion(tree, motion.fps(), motion.motionName(),
                                     std::move(roots), std::move(rotations));
}

ArticulationMotionMappingResult ArticulationMotionMapper::toArticulationMotion(
    const SkeletonMotion& source, bool clampToLimits) const {
    const int frames = source.numFrames();
    const int joints = source.numJoints();
    std::vector<float> q(static_cast<size_t>(frames) * _layout.nq());
    std::vector<float> residuals(static_cast<size_t>(frames) * joints);

    for (int frame = 0; frame < frames; ++frame) {
        ArticulationMappingResult mapped =
            toArticulationCoordinates(source.frame(frame), clampToLimits);
        std::copy(mapped.q.begin(), mapped.q.end(),
                  q.begin() + static_cast<size_t>(frame) * _layout.nq());
        std::copy(mapped.residualAngles.begin(), mapped.residualAngles.end(),
                  residuals.begin() + static_cast<size_t>(frame) * joints);
    }

    for (const auto& block : _layout.blocks()) {
        if (block.type != ArticulationCoordinateType::Free &&
            block.type != ArticulationCoordinateType::Spherical)
            continue;
        const int rotationOffset =
            block.qOffset +
            (block.type == ArticulationCoordinateType::Free ? 3 : 0);
        for (int frame = 1; frame < frames; ++frame) {
            float* previous = q.data() +
                              static_cast<size_t>(frame - 1) * _layout.nq() +
                              rotationOffset;
            float* current = q.data() +
                             static_cast<size_t>(frame) * _layout.nq() +
                             rotationOffset;
            const float dot =
                previous[0] * current[0] + previous[1] * current[1] +
                previous[2] * current[2] + previous[3] * current[3];
            if (dot < 0.0f)
                for (int component = 0; component < 4; ++component)
                    current[component] = -current[component];
        }
    }

    std::vector<float> qd(static_cast<size_t>(frames) * _layout.nv(), 0.0f);
    auto quaternionAt = [&](int frame, int offset) {
        const float* value =
            q.data() + static_cast<size_t>(frame) * _layout.nq() + offset;
        return normalized(
            Eigen::Quaternionf(value[0], value[1], value[2], value[3]));
    };
    for (int frame = 0; frame < frames; ++frame) {
        if (frames <= 1)
            break;
        const int before = frame == 0 ? 0 : frame - 1;
        const int after = frame == frames - 1 ? frames - 1 : frame + 1;
        const float scale = source.fps() / static_cast<float>(after - before);
        for (const auto& block : _layout.blocks()) {
            float* velocity = qd.data() +
                              static_cast<size_t>(frame) * _layout.nv() +
                              block.qdOffset;
            if (block.type == ArticulationCoordinateType::Free) {
                const float* p0 = q.data() +
                                  static_cast<size_t>(before) * _layout.nq() +
                                  block.qOffset;
                const float* p1 = q.data() +
                                  static_cast<size_t>(after) * _layout.nq() +
                                  block.qOffset;
                for (int axis = 0; axis < 3; ++axis)
                    velocity[axis] = (p1[axis] - p0[axis]) * scale;
                const Eigen::Quaternionf q0 =
                    quaternionAt(before, block.qOffset + 3);
                const Eigen::Quaternionf q1 =
                    quaternionAt(after, block.qOffset + 3);
                const Eigen::Vector3f omega =
                    rotationVector(q1 * q0.conjugate()) * scale;
                velocity[3] = omega.x();
                velocity[4] = omega.y();
                velocity[5] = omega.z();
            } else if (block.type == ArticulationCoordinateType::Spherical) {
                const Eigen::Quaternionf q0 =
                    quaternionAt(before, block.qOffset);
                const Eigen::Quaternionf q1 =
                    quaternionAt(after, block.qOffset);
                const Eigen::Vector3f omega =
                    rotationVector(q1 * q0.conjugate()) * scale;
                velocity[0] = omega.x();
                velocity[1] = omega.y();
                velocity[2] = omega.z();
            } else if (block.type == ArticulationCoordinateType::Revolute) {
                const float a0 = q[static_cast<size_t>(before) * _layout.nq() +
                                   block.qOffset];
                const float a1 = q[static_cast<size_t>(after) * _layout.nq() +
                                   block.qOffset];
                constexpr float twoPi = 6.28318530717958647692f;
                velocity[0] = std::remainder(a1 - a0, twoPi) * scale;
            } else if (block.type == ArticulationCoordinateType::Prismatic) {
                const float p0 = q[static_cast<size_t>(before) * _layout.nq() +
                                   block.qOffset];
                const float p1 = q[static_cast<size_t>(after) * _layout.nq() +
                                   block.qOffset];
                velocity[0] = (p1 - p0) * scale;
            }
        }
    }

    return {ArticulationMotion(_layout, frames, source.fps(),
                               source.motionName(), std::move(q),
                               std::move(qd)),
            std::move(residuals)};
}

} // namespace KE::Animation

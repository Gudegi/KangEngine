#include "skeleton_motion.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace KE {
namespace Animation {

SkeletonMotion::SkeletonMotion(std::shared_ptr<const SkeletonTree> tree,
                               float fps, std::string motionName,
                               std::vector<float> rootTranslations,
                               std::vector<float> localRotationsWxyz)
    : _fps(fps), _tree(std::move(tree)), _motionName(std::move(motionName)),
      _rootTranslations(std::move(rootTranslations)),
      _localRotationsWxyz(std::move(localRotationsWxyz)) {
    if (!_tree)
        throw std::runtime_error("SkeletonMotion requires a SkeletonTree");
    if (_fps <= 0.0f)
        throw std::runtime_error("SkeletonMotion fps must be positive");
    if (_rootTranslations.size() % 3 != 0)
        throw std::runtime_error(
            "SkeletonMotion rootTranslations must have shape [frames, 3]");

    const int frames = numFrames();
    const int joints = numJoints();
    if (static_cast<int>(_localRotationsWxyz.size()) != frames * joints * 4) {
        throw std::runtime_error(
            "SkeletonMotion localRotationsWxyz must have shape "
            "[frames, joints, 4]");
    }
}

int SkeletonMotion::numFrames() const {
    return static_cast<int>(_rootTranslations.size() / 3);
}

float SkeletonMotion::duration() const {
    if (numFrames() <= 0)
        return 0.0f;
    return static_cast<float>(numFrames() - 1) / _fps;
}

Eigen::Vector3f SkeletonMotion::rootTranslation(int frameIndex) const {
    if (frameIndex < 0 || frameIndex >= numFrames())
        throw std::out_of_range("SkeletonMotion root frame out of range");
    const float* p = _rootTranslations.data() + frameIndex * 3;
    return Eigen::Vector3f(p[0], p[1], p[2]);
}

Eigen::Quaternionf SkeletonMotion::localRotation(int frameIndex,
                                                 int jointIndex) const {
    if (frameIndex < 0 || frameIndex >= numFrames())
        throw std::out_of_range("SkeletonMotion rotation frame out of range");
    if (jointIndex < 0 || jointIndex >= numJoints())
        throw std::out_of_range("SkeletonMotion rotation joint out of range");

    const size_t offset =
        (static_cast<size_t>(frameIndex) * static_cast<size_t>(numJoints()) +
         static_cast<size_t>(jointIndex)) *
        4;
    const float* q = _localRotationsWxyz.data() + offset;
    Eigen::Quaternionf quat(q[0], q[1], q[2], q[3]);
    if (quat.norm() <= 1e-6f)
        return Eigen::Quaternionf::Identity();
    return quat.normalized();
}

SkeletonState SkeletonMotion::frame(int frameIndex) const {
    std::vector<Eigen::Quaternionf> rotations;
    rotations.reserve(static_cast<size_t>(numJoints()));
    for (int j = 0; j < numJoints(); ++j)
        rotations.push_back(localRotation(frameIndex, j));

    return SkeletonState::fromRotationAndRootTranslation(
        _tree, rotations, rootTranslation(frameIndex), true);
}

SkeletonState SkeletonMotion::sample(float time, bool loop) const {
    const int frames = numFrames();
    if (frames <= 0)
        throw std::runtime_error("Cannot sample empty SkeletonMotion");
    if (frames == 1)
        return frame(0);

    const float maxTime = duration();
    float t = time;
    if (loop && maxTime > 0.0f) {
        t = std::fmod(t, maxTime);
        if (t < 0.0f)
            t += maxTime;
    } else {
        t = std::clamp(t, 0.0f, maxTime);
    }

    const float frameFloat = t * _fps;
    const int i0 =
        std::clamp(static_cast<int>(std::floor(frameFloat)), 0, frames - 1);
    const int i1 = std::min(i0 + 1, frames - 1);
    const float alpha = frameFloat - static_cast<float>(i0);

    const Eigen::Vector3f root =
        rootTranslation(i0) * (1.0f - alpha) + rootTranslation(i1) * alpha;

    std::vector<Eigen::Quaternionf> rotations;
    rotations.reserve(static_cast<size_t>(numJoints()));
    for (int j = 0; j < numJoints(); ++j) {
        Eigen::Quaternionf q0 = localRotation(i0, j);
        Eigen::Quaternionf q1 = localRotation(i1, j);
        if (q0.dot(q1) < 0.0f)
            q1.coeffs() *= -1.0f;
        rotations.push_back(q0.slerp(alpha, q1).normalized());
    }

    return SkeletonState::fromRotationAndRootTranslation(_tree, rotations, root,
                                                         true);
}

} // namespace Animation
} // namespace KE

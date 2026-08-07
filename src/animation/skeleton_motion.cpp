#include "skeleton_motion.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace KE {
namespace Animation {

namespace {

std::vector<Eigen::Vector3f>
differentiateVectors(const std::vector<Eigen::Vector3f>& values, int frames,
                     int width, float fps) {
    std::vector<Eigen::Vector3f> output(values.size(), Eigen::Vector3f::Zero());
    if (frames <= 1)
        return output;
    for (int frameIndex = 0; frameIndex < frames; ++frameIndex) {
        const int before = frameIndex == 0 ? 0 : frameIndex - 1;
        const int after =
            frameIndex == frames - 1 ? frames - 1 : frameIndex + 1;
        const float scale = fps / static_cast<float>(after - before);
        for (int item = 0; item < width; ++item) {
            output[static_cast<size_t>(frameIndex * width + item)] =
                (values[static_cast<size_t>(after * width + item)] -
                 values[static_cast<size_t>(before * width + item)]) *
                scale;
        }
    }
    return output;
}

Eigen::Vector3f angularDifference(const Eigen::Quaternionf& before,
                                  const Eigen::Quaternionf& after,
                                  float inverseDeltaTime) {
    Eigen::Quaternionf delta = after * before.conjugate();
    if (delta.w() < 0.0f)
        delta.coeffs() *= -1.0f;
    if (delta.norm() <= 1e-8f)
        return Eigen::Vector3f::Zero();
    delta.normalize();
    Eigen::AngleAxisf angleAxis(delta);
    if (!std::isfinite(angleAxis.angle()))
        return Eigen::Vector3f::Zero();
    return angleAxis.axis() * angleAxis.angle() * inverseDeltaTime;
}

} // namespace

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
    const float loopPeriod = static_cast<float>(frames) / _fps;
    float t = time;
    if (loop && loopPeriod > 0.0f) {
        t = std::fmod(t, loopPeriod);
        if (t < 0.0f)
            t += loopPeriod;
    } else {
        t = std::clamp(t, 0.0f, maxTime);
    }

    const float frameFloat = t * _fps;
    const int i0 =
        std::clamp(static_cast<int>(std::floor(frameFloat)), 0, frames - 1);
    const int i1 = std::min(i0 + 1, frames - 1);
    const float alpha = i0 == i1 ? 0.0f : frameFloat - static_cast<float>(i0);

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

std::vector<Transform> SkeletonMotion::globalTransforms() const {
    std::vector<Transform> output(static_cast<size_t>(numFrames()) *
                                  static_cast<size_t>(numJoints()));
    std::vector<Transform> frameTransforms;
    for (int frameIndex = 0; frameIndex < numFrames(); ++frameIndex) {
        frame(frameIndex).computeGlobalTransformsInto(frameTransforms);
        std::copy(frameTransforms.begin(), frameTransforms.end(),
                  output.begin() + static_cast<size_t>(frameIndex) *
                                       static_cast<size_t>(numJoints()));
    }
    return output;
}

std::vector<Eigen::Vector3f> SkeletonMotion::globalPositions() const {
    const auto transforms = globalTransforms();
    std::vector<Eigen::Vector3f> output;
    output.reserve(transforms.size());
    for (const Transform& transform : transforms)
        output.push_back(transform.translation);
    return output;
}

std::vector<Eigen::Vector3f> SkeletonMotion::globalLinearVelocities() const {
    return differentiateVectors(globalPositions(), numFrames(), numJoints(),
                                fps());
}

std::vector<Eigen::Vector3f> SkeletonMotion::globalAngularVelocities() const {
    const auto transforms = globalTransforms();
    std::vector<Eigen::Vector3f> output(transforms.size(),
                                        Eigen::Vector3f::Zero());
    if (numFrames() <= 1)
        return output;
    for (int frameIndex = 0; frameIndex < numFrames(); ++frameIndex) {
        const int before = frameIndex == 0 ? 0 : frameIndex - 1;
        const int after =
            frameIndex == numFrames() - 1 ? numFrames() - 1 : frameIndex + 1;
        const float scale = fps() / static_cast<float>(after - before);
        for (int joint = 0; joint < numJoints(); ++joint) {
            output[static_cast<size_t>(frameIndex * numJoints() + joint)] =
                angularDifference(
                    transforms[static_cast<size_t>(before * numJoints() +
                                                   joint)]
                        .rotation,
                    transforms[static_cast<size_t>(after * numJoints() + joint)]
                        .rotation,
                    scale);
        }
    }
    return output;
}

std::vector<Eigen::Vector3f> SkeletonMotion::globalLinearAccelerations() const {
    return differentiateVectors(globalLinearVelocities(), numFrames(),
                                numJoints(), fps());
}

std::vector<Eigen::Vector3f>
SkeletonMotion::globalAngularAccelerations() const {
    return differentiateVectors(globalAngularVelocities(), numFrames(),
                                numJoints(), fps());
}

std::vector<Eigen::Vector3f> SkeletonMotion::rootLinearVelocities() const {
    std::vector<Eigen::Vector3f> roots;
    roots.reserve(static_cast<size_t>(numFrames()));
    for (int frameIndex = 0; frameIndex < numFrames(); ++frameIndex)
        roots.push_back(rootTranslation(frameIndex));
    return differentiateVectors(roots, numFrames(), 1, fps());
}

std::vector<Eigen::Vector3f> SkeletonMotion::rootLinearAccelerations() const {
    return differentiateVectors(rootLinearVelocities(), numFrames(), 1, fps());
}

} // namespace Animation
} // namespace KE

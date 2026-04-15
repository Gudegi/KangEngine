#include "skeleton_fk.hpp"

namespace KE {
namespace Animation {

SkeletonFK SkeletonFK::fromMJCF(const std::string& mjcfPath, float scale,
                                const std::string& order) {
    return fromData(MJCFLoader::load(mjcfPath, 1.0f, order), scale);
}

SkeletonFK SkeletonFK::fromData(const MJCFData& data, float scale) {
    SkeletonFK fk;
    fk._scale = scale;
    fk._skeleton = data.skeletonTree;
    fk._state = SkeletonState::zeroPose(fk._skeleton);
    return fk;
}

void SkeletonFK::setJointRotation(int idx, const Eigen::Quaternionf& q) {
    _state.setRotation(idx, q);
}

void SkeletonFK::setRootTranslation(const Eigen::Vector3f& t) {
    _state.setRootTranslation(t);
}

void SkeletonFK::resetToZeroPose() {
    _state = SkeletonState::zeroPose(_skeleton);
}

} // namespace Animation
} // namespace KE

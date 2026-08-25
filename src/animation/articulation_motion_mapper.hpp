#pragma once

#include "animation/skeleton_motion.hpp"
#include "animation/skeleton_state.hpp"
#include "animation/articulation_motion.hpp"

#include <vector>

namespace KE::Animation {

struct ArticulationMappingResult {
    std::vector<float> q;
    std::vector<float> residualAngles;
};

struct ArticulationMotionMappingResult {
    ArticulationMotion motion;
    std::vector<float> residualAngles;
};

class ArticulationMotionMapper {
  public:
    explicit ArticulationMotionMapper(ArticulationCoordinateLayout layout);

    // Convert one local SkeletonState into canonical articulation q.
    ArticulationMappingResult
    toArticulationCoordinates(const SkeletonState& state,
                              bool clampToLimits = false) const;

    // Reconstruct one local SkeletonState from canonical articulation q.
    SkeletonState toSkeletonState(const std::vector<float>& q) const;

    // Reconstruct a SkeletonMotion from canonical articulation q frames.
    SkeletonMotion toSkeletonMotion(const ArticulationMotion& motion) const;

    // Convert a SkeletonMotion into canonical q/qd and frame residuals.
    ArticulationMotionMappingResult
    toArticulationMotion(const SkeletonMotion& motion,
                         bool clampToLimits = false) const;

    // Return the coordinate layout used by this mapper.
    const ArticulationCoordinateLayout& layout() const { return _layout; }

  private:
    ArticulationCoordinateLayout _layout;
};

} // namespace KE::Animation

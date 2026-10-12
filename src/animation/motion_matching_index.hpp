#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace KE::Animation {

// Candidates start at frame 0; the tail without a full future horizon is excluded.
struct MotionMatchingClipInfo {
    std::size_t frameCount;
    std::size_t candidateCount;
    double fps;
};

struct MotionMatch {
    std::size_t row;
    std::size_t motionIndex;
    std::size_t frameIndex;
    double time; // Clip-local seconds.
    double cost; // Squared distance minus continuation bias; may be negative.
};

struct MotionMatchingSearchOptions {
    std::optional<std::size_t> currentMotionIndex;
    std::optional<double> currentTime;
    double continuationBias = 0.0;
    // nullopt selects all clips; empty is invalid. Ties favor the lowest row.
    std::optional<std::vector<std::size_t>> allowedMotionIndices;
};

// Owns normalized, weighted row-major features. Safe for concurrent searches.
class MotionMatchingIndex {
  public:
    MotionMatchingIndex(std::vector<double> features,
                        std::size_t featureDimension,
                        std::vector<MotionMatchingClipInfo> clips);

    MotionMatch search(const std::vector<double>& query,
                       const MotionMatchingSearchOptions& options = {}) const;

    std::size_t numFrames() const { return _rowOffsets.back(); }
    std::size_t featureDimension() const { return _featureDimension; }
    std::size_t numMotions() const { return _clips.size(); }

  private:
    std::vector<double> _features;
    std::size_t _featureDimension;
    std::vector<MotionMatchingClipInfo> _clips;
    std::vector<std::size_t> _rowOffsets;
};

} // namespace KE::Animation

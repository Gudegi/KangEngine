#include "motion_matching_index.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace KE::Animation {

MotionMatchingIndex::MotionMatchingIndex(
    std::vector<double> features, std::size_t featureDimension,
    std::vector<MotionMatchingClipInfo> clips)
    : _features(std::move(features)), _featureDimension(featureDimension),
      _clips(std::move(clips)), _rowOffsets{0} {
    if (_featureDimension == 0 || _clips.empty())
        throw std::invalid_argument("features and clips must be nonempty");
    for (const auto& clip : _clips) {
        if (clip.frameCount < 2 || clip.candidateCount == 0 ||
            clip.candidateCount >= clip.frameCount ||
            !std::isfinite(clip.fps) || clip.fps <= 0 ||
            !std::isfinite(static_cast<double>(clip.frameCount - 1) / clip.fps))
            throw std::invalid_argument("invalid clip frame counts or fps");
        if (clip.candidateCount >
            std::numeric_limits<std::size_t>::max() - _rowOffsets.back())
            throw std::invalid_argument("candidate count overflow");
        _rowOffsets.push_back(_rowOffsets.back() + clip.candidateCount);
    }
    if (numFrames() >
            std::numeric_limits<std::size_t>::max() / _featureDimension ||
        _features.size() != numFrames() * _featureDimension)
        throw std::invalid_argument(
            "feature shape does not match clip candidates");
    for (double value : _features)
        if (!std::isfinite(value))
            throw std::invalid_argument("features must be finite");
}

MotionMatch
MotionMatchingIndex::search(const std::vector<double>& query,
                            const MotionMatchingSearchOptions& options) const {
    if (query.size() != _featureDimension ||
        !std::all_of(query.begin(), query.end(),
                     [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            "query must have feature_dimension finite values");
    if (!std::isfinite(options.continuationBias) ||
        options.continuationBias < 0)
        throw std::invalid_argument(
            "continuation_bias must be finite and nonnegative");
    if (options.currentMotionIndex.has_value() !=
        options.currentTime.has_value())
        throw std::invalid_argument(
            "current_motion_index and current_time must be supplied together");
    if (options.continuationBias != 0 && !options.currentMotionIndex)
        throw std::invalid_argument(
            "continuation_bias requires current motion and time");

    std::size_t continuationRow = numFrames();
    if (options.currentMotionIndex) {
        const auto motion = *options.currentMotionIndex;
        if (motion >= _clips.size())
            throw std::invalid_argument(
                "current_motion_index is outside the database");
        const auto& clip = _clips[motion];
        const double time = *options.currentTime;
        const double lastFrame = static_cast<double>(clip.frameCount - 1);
        if (!std::isfinite(time) || time < 0 ||
            time > lastFrame / clip.fps + 1e-7)
            throw std::invalid_argument(
                "current_time is outside the sampled clip interval");
        const double frame =
            std::floor(std::min(time * clip.fps, lastFrame) + 0.5 + 1e-7);
        if (frame < static_cast<double>(clip.candidateCount))
            continuationRow =
                _rowOffsets[motion] + static_cast<std::size_t>(frame);
    }
    if (options.allowedMotionIndices) {
        if (options.allowedMotionIndices->empty())
            throw std::invalid_argument(
                "allowed_motion_indices must select at least one motion");
        for (auto motion : *options.allowedMotionIndices)
            if (motion >= _clips.size())
                throw std::invalid_argument(
                    "allowed_motion_indices contains an invalid motion");
    }

    MotionMatch best{0, 0, 0, 0.0, std::numeric_limits<double>::infinity()};
    for (std::size_t motion = 0; motion < _clips.size(); ++motion) {
        if (options.allowedMotionIndices &&
            std::find(options.allowedMotionIndices->begin(),
                      options.allowedMotionIndices->end(),
                      motion) == options.allowedMotionIndices->end())
            continue;
        for (std::size_t row = _rowOffsets[motion];
             row < _rowOffsets[motion + 1]; ++row) {
            const double* feature = _features.data() + row * _featureDimension;
            double cost = 0.0;
            for (std::size_t dimension = 0; dimension < _featureDimension;
                 ++dimension) {
                const double difference = feature[dimension] - query[dimension];
                cost += difference * difference;
            }
            if (row == continuationRow)
                cost -= options.continuationBias;
            if (cost < best.cost) {
                const auto frame = row - _rowOffsets[motion];
                best = {row, motion, frame,
                        static_cast<double>(frame) / _clips[motion].fps, cost};
            }
        }
    }
    if (!std::isfinite(best.cost))
        throw std::invalid_argument(
            "query magnitude produced nonfinite squared distances");
    return best;
}

} // namespace KE::Animation

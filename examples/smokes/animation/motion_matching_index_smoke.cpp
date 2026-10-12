// Standalone CPU smoke; no renderer or Python runtime required.
#include "animation/motion_matching_index.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace KE::Animation;

int main() {
    // Two normalized 2D features per clip. Both clips contain excluded tails.
    MotionMatchingIndex index({0, 0, 1, 0, 0, 0, 0, 2}, 2,
                              {{4, 2, 30}, {5, 2, 60}});
    auto require = [](bool condition) {
        if (!condition)
            throw std::runtime_error("motion matching index smoke failed");
    };
    require(index.search({0, 0}).row == 0);
    MotionMatchingSearchOptions options;
    options.allowedMotionIndices = std::vector<std::size_t>{1};
    auto result = index.search({0, 1.8}, options);
    require(result.motionIndex == 1 && result.frameIndex == 1);
    require(result.time == 1.0 / 60 && std::abs(result.cost - 0.04) < 1e-12);
    options.currentMotionIndex = 1;
    options.currentTime = 0;
    options.continuationBias = 1;
    require(index.search({0, 0}, options).cost == -1);
    options.currentTime = 4.0 / 60; // Playback tail must not receive a bias.
    require(index.search({0, 0}, options).cost == 0);
    options.allowedMotionIndices = std::vector<std::size_t>{};
    bool rejected = false;
    try {
        index.search({0, 0}, options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected);
    std::cout << "MotionMatchingIndex smoke passed\n";
}

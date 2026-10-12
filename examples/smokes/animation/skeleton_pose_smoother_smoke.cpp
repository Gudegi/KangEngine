// A procedural pose source demonstrates the same boundary a decoder will use.
// No SkeletonMotion, matching index, Python, or renderer is needed.
#include "animation/skeleton_pose_smoother.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace KE::Animation;

int main() {
    auto tree = std::make_shared<SkeletonTree>(
        std::vector<std::string>{"root", "hand"}, std::vector<int>{-1, 0},
        std::vector<Eigen::Vector3f>{{0, 0, 0}, {0, 0, 1}},
        std::vector<Eigen::Quaternionf>(2, Eigen::Quaternionf::Identity()),
        std::vector<int>{1, 1});
    SkeletonPoseSample target{
        SkeletonState::zeroPose(tree), Eigen::Vector3f::UnitX(),
        std::vector<Eigen::Vector3f>(2, Eigen::Vector3f::Zero())};
    SkeletonPoseSmoother smoother(target, .2f);
    target.state.setRootTranslation({0, 0, 1});
    smoother.transition(target);
    const auto* storage = smoother.sample().state.rotations().data();
    const double dt = 1.0 / 60;
    for (int frame = 1; frame <= 120; ++frame) {
        const double time = frame * dt;
        target.state.setRootTranslation({static_cast<float>(time), 0, 1});
        smoother.advance(target, dt);
        const auto& output = smoother.sample();
        const double lambda = 2 * std::log(2.) / smoother.halfLife();
        const double height =
            1 - (1 + lambda * time) * std::exp(-lambda * time);
        if ((output.state.rootTranslation() - Eigen::Vector3f(time, 0, height))
                    .norm() > 2e-5 ||
            output.state.rotations().data() != storage)
            throw std::runtime_error(
                "procedural pose transition or storage reuse failed");
    }
    std::cout << "SkeletonPoseSmoother smoke passed\n";
}

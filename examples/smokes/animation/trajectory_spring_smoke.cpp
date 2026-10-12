#include "animation/trajectory_spring.hpp"
#include <cfenv>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace KE::Animation;
    const Eigen::Vector3d p(1, 2, 3), v(.2, .3, .4), a(.1, -.2, .3), goal(2, 0, 1);
    const auto whole = trajectorySpringPosition(p, v, a, goal, .2, .5);
    auto split = trajectorySpringPosition(p, v, a, goal, .2, .2);
    split = trajectorySpringPosition(std::get<0>(split), std::get<1>(split),
                                      std::get<2>(split), goal, .2, .3);
    if ((std::get<0>(whole) - std::get<0>(split)).norm() > 1e-12 ||
        (std::get<1>(whole) - std::get<1>(split)).norm() > 1e-12)
        throw std::runtime_error("position substeps differ");
    const Eigen::Quaterniond target(Eigen::AngleAxisd(.7, Eigen::Vector3d::UnitY()));
    const auto rotation = trajectorySpringRotation(Eigen::Quaterniond::Identity(),
        Eigen::Vector3d::Zero(), target, .2, 1000);
    if (std::abs(rotation.first.dot(target)) < 1 - 1e-12 || rotation.second.norm() != 0)
        throw std::runtime_error("rotation did not settle");
    volatile double halfLife = 0;
    std::feclearexcept(FE_ALL_EXCEPT);
    const auto snapped = trajectorySpringPosition(p, v, a, goal, halfLife, .1);
    if (std::fetestexcept(FE_DIVBYZERO | FE_INVALID) ||
        (std::get<0>(snapped) - (p + goal * .1)).norm() > 1e-12 ||
        (std::get<1>(snapped) - goal).norm() > 1e-12 ||
        std::get<2>(snapped).norm() != 0)
        throw std::runtime_error("zero half-life position must snap without division by zero");
    std::feclearexcept(FE_ALL_EXCEPT);
    const auto snappedRotation = trajectorySpringRotation(
        Eigen::Quaterniond::Identity(), v, target, halfLife, .1);
    if (std::fetestexcept(FE_DIVBYZERO | FE_INVALID) ||
        std::abs(snappedRotation.first.dot(target)) < 1 - 1e-12 ||
        snappedRotation.second.norm() != 0)
        throw std::runtime_error("zero half-life rotation must snap without division by zero");
    std::cout << "trajectory springs passed\n";
}

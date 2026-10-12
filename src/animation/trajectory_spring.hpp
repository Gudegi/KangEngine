// Closed-form critically damped trajectory springs, based on Daniel Holden's
// GenoViewPython-MotionMatching (TrajectorySpringPosition/Rotation):
// https://github.com/orangeduck/GenoViewPython-MotionMatching

#pragma once

#include "quaternion_math.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace KE::Animation {

namespace trajectory_spring_detail {
constexpr double kTwoLogTwo = 1.3862943611198906188;

inline void validateTime(double halfLife, double dt) {
    if (!std::isfinite(halfLife) || halfLife < 0 || !std::isfinite(dt) ||
        dt < 0)
        throw std::invalid_argument(
            "half_life and dt must be finite and nonnegative");
}

inline void validateVector(const Eigen::Vector3d& value) {
    if (!value.allFinite())
        throw std::invalid_argument("spring vectors must be finite");
}

} // namespace trajectory_spring_detail

// Returns (position, velocity, acceleration) after dt at constant desiredVelocity.
// dt=0 preserves state; otherwise halfLife=0 snaps velocity and clears acceleration.
inline std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
trajectorySpringPosition(const Eigen::Vector3d& position,
                         const Eigen::Vector3d& velocity,
                         const Eigen::Vector3d& acceleration,
                         const Eigen::Vector3d& desiredVelocity,
                         double halfLife, double dt) {
    using namespace trajectory_spring_detail;
    validateTime(halfLife, dt);
    for (const auto* v :
         {&position, &velocity, &acceleration, &desiredVelocity})
        validateVector(*v);
    if (dt == 0)
        return {position, velocity, acceleration};
    Eigen::Vector3d p, v = desiredVelocity, a = Eigen::Vector3d::Zero();
    const double tau = halfLife / kTwoLogTwo;
    if (tau == 0) {
        p = position + desiredVelocity * dt;
        validateVector(p);
        return {p, v, a};
    }
    const double s = dt / tau;
    const Eigen::Vector3d error = velocity - desiredVelocity;
    if (s > 700) {
        p = position + desiredVelocity * dt +
            tau * (2 * error + tau * acceleration);
    } else {
        const double e = std::exp(-s);
        // Integral coefficients use series near zero to avoid cancellation.
        const double f1 = s < 1e-4 ? 1 - s / 2 + s * s / 6 - s * s * s / 24
                                   : -std::expm1(-s) / s;
        const double f2 = s < 1e-4 ? 0.5 - s / 3 + s * s / 8 - s * s * s / 30
                                   : (-std::expm1(-s) - s * e) / (s * s);
        p = position + desiredVelocity * dt +
            dt * ((f1 + s * f2) * error + (dt * f2) * acceleration);
        v += (e * (1 + s)) * error + (e * dt) * acceleration;
        a = (e * (1 - s)) * acceleration - ((e * s) * error) / tau;
    }
    validateVector(p);
    validateVector(v);
    validateVector(a);
    return {p, v, a};
}

// Returns (rotation, spatial angular velocity) after dt toward constant desiredRotation.
// Angular velocity uses the input frame; the shortest rotation can switch at pi.
inline std::pair<Eigen::Quaterniond, Eigen::Vector3d> trajectorySpringRotation(
    const Eigen::Quaterniond& rotation, const Eigen::Vector3d& angularVelocity,
    const Eigen::Quaterniond& desiredRotation, double halfLife, double dt) {
    using namespace trajectory_spring_detail;
    validateTime(halfLife, dt);
    validateVector(angularVelocity);
    const auto q = unitQuaternion(rotation);
    const auto goal = unitQuaternion(desiredRotation);
    if (dt == 0)
        return {q, angularVelocity};
    const double tau = halfLife / kTwoLogTwo;
    if (tau == 0)
        return {goal, Eigen::Vector3d::Zero()};
    const double s = dt / tau;
    if (s > 700)
        return {goal, Eigen::Vector3d::Zero()};
    Eigen::Quaterniond difference = q * goal.conjugate();
    difference.w() = q.dot(goal);
    const Eigen::Vector3d error = quatToRotationVector(difference);
    const Eigen::Vector3d derivative = inverseLeftJacobian(error, angularVelocity);
    const double e = std::exp(-s);
    const Eigen::Vector3d r = (e * (1 + s)) * error + (e * dt) * derivative;
    const Eigen::Vector3d dr =
        (e * (1 - s)) * derivative - ((e * s) * error) / tau;
    validateVector(r);
    validateVector(dr);
    const Eigen::Quaterniond result = quatFromRotationVector(r) * goal;
    const Eigen::Vector3d velocity = leftJacobian(r, dr);
    validateVector(velocity);
    return {unitQuaternion(result), velocity};
}

} // namespace KE::Animation

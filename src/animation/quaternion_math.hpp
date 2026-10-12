#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace KE::Animation {

// Rejects non-unit input; normalizes only small numerical drift.
template <typename Scalar>
Eigen::Quaternion<Scalar> unitQuaternion(const Eigen::Quaternion<Scalar>& quat) {
    if (!quat.coeffs().allFinite() || std::abs(quat.norm() - Scalar(1)) > Scalar(1e-3))
        throw std::invalid_argument("quaternion must be finite and unit length");
    return quat.normalized();
}

// Aligns target's sign to reference; at pi, prefers a positive relative axis.
template <typename Scalar>
Eigen::Quaternion<Scalar> alignQuaternion(
    const Eigen::Quaternion<Scalar>& reference, Eigen::Quaternion<Scalar> target) {
    const Scalar dot = reference.dot(target);
    bool negate = dot < 0;
    if (dot == 0) {
        const Eigen::Quaternion<Scalar> difference = target * reference.conjugate();
        for (int axis = 0; axis < 3; ++axis) {
            if (difference.vec()[axis] != 0) {
                negate = difference.vec()[axis] < 0;
                break;
            }
        }
    }
    if (negate)
        target.coeffs() *= -1;
    return target;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> quatToRotationVector(Eigen::Quaternion<Scalar> quat) {
    quat = alignQuaternion(Eigen::Quaternion<Scalar>::Identity(), quat.normalized());
    const double length = quat.vec().template cast<double>().norm();
    if (length < 1e-8)
        return Scalar(2) * quat.vec();
    return quat.vec() * Scalar(2 * std::atan2(length, quat.w()) / length);
}

template <typename Scalar>
Eigen::Quaternion<Scalar> quatFromRotationVector(
    const Eigen::Matrix<Scalar, 3, 1>& rotationVector) {
    const double angle = rotationVector.template cast<double>().norm();
    const double scale = angle < 1e-8 ? 0.5 : std::sin(angle / 2) / angle;
    return Eigen::Quaternion<Scalar>(Scalar(std::cos(angle / 2)),
                                      Scalar(scale) * rotationVector.x(),
                                      Scalar(scale) * rotationVector.y(),
                                      Scalar(scale) * rotationVector.z()).normalized();
}

// Converts rotation-vector derivatives to spatial angular velocity.
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> leftJacobian(
    const Eigen::Matrix<Scalar, 3, 1>& rotationVector,
    const Eigen::Matrix<Scalar, 3, 1>& derivative) {
    const double angleSquared = rotationVector.template cast<double>().squaredNorm();
    double a, b;
    if (angleSquared < 1e-6) {
        a = 0.5 - angleSquared / 24;
        b = 1.0 / 6 - angleSquared / 120;
    } else {
        const double angle = std::sqrt(angleSquared);
        a = (1 - std::cos(angle)) / angleSquared;
        b = (angle - std::sin(angle)) / (angleSquared * angle);
    }
    return derivative + Scalar(a) * rotationVector.cross(derivative) +
           Scalar(b) * rotationVector.cross(rotationVector.cross(derivative));
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> inverseLeftJacobian(
    const Eigen::Matrix<Scalar, 3, 1>& rotationVector,
    const Eigen::Matrix<Scalar, 3, 1>& velocity) {
    const double angleSquared = rotationVector.template cast<double>().squaredNorm();
    double c;
    if (angleSquared < 1e-6) {
        c = 1.0 / 12 + angleSquared / 720;
    } else {
        const double angle = std::sqrt(angleSquared);
        c = (1 - 0.5 * angle / std::tan(0.5 * angle)) / angleSquared;
    }
    return velocity - Scalar(0.5) * rotationVector.cross(velocity) +
           Scalar(c) * rotationVector.cross(rotationVector.cross(velocity));
}

} // namespace KE::Animation

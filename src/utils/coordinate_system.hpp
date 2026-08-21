#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <stdexcept>
#include <string>
#include <string_view>

namespace KE::Utils {

// Shared coordinate-system conversion utilities. The reference basis is an
// internal calculation frame and does not impose an engine-wide coordinate
// policy.

enum class CoordinateSystem {
    YUpZForward,
    YUpNegativeZForward,
    ZUpXForward,
};

inline Eigen::Matrix3f coordinateReferenceBasis(CoordinateSystem system) {
    switch (system) {
    case CoordinateSystem::YUpZForward:
        return Eigen::Matrix3f::Identity();
    case CoordinateSystem::YUpNegativeZForward:
        return (Eigen::Matrix3f() << -1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
                -1.f)
            .finished();
    case CoordinateSystem::ZUpXForward:
        return (Eigen::Matrix3f() << 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f,
                0.f)
            .finished();
    }
    throw std::invalid_argument("unknown coordinate system");
}

inline Eigen::Matrix3f coordinateConversionMatrix(CoordinateSystem source,
                                                  CoordinateSystem target) {
    return coordinateReferenceBasis(target).transpose() *
           coordinateReferenceBasis(source);
}

inline Eigen::Quaternionf coordinateSystemRotation(CoordinateSystem source,
                                                   CoordinateSystem target) {
    return Eigen::Quaternionf(coordinateConversionMatrix(source, target))
        .normalized();
}

inline CoordinateSystem coordinateSystemFromString(std::string_view value) {
    if (value == "y_up_z_forward")
        return CoordinateSystem::YUpZForward;
    if (value == "y_up_neg_z_forward")
        return CoordinateSystem::YUpNegativeZForward;
    if (value == "z_up_x_forward")
        return CoordinateSystem::ZUpXForward;
    throw std::invalid_argument("unknown coordinate system: " +
                                std::string(value));
}

} // namespace KE::Utils

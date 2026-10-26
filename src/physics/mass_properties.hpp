#pragma once

#include <PxPhysicsAPI.h>
#include <cmath>
#include <stdexcept>

namespace KE::Physics {

inline void validateMass(float mass) {
    if (!std::isfinite(mass) || mass <= 0.f)
        throw std::invalid_argument("mass must be finite and positive");
}

inline void validateMassSpaceInertia(const physx::PxVec3& inertia) {
    if (!inertia.isFinite() || inertia.x <= 0.f || inertia.y <= 0.f ||
        inertia.z <= 0.f)
        throw std::invalid_argument("mass-space inertia must be finite and positive");
}

inline void validateLocalCOMPose(const physx::PxTransform& pose) {
    if (!pose.isValid())
        throw std::invalid_argument(
            "local COM pose requires finite values and a unit quaternion");
}

} // namespace KE::Physics

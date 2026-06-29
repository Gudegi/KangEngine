#ifndef _PHYSX_COMPAT_HPP_
#define _PHYSX_COMPAT_HPP_

#include <PxArticulationJointReducedCoordinate.h>
#include <foundation/PxPhysicsVersion.h>

#define KANGENGINE_PHYSX_VERSION_AT_LEAST(major, minor)                      \
    ((PX_PHYSICS_VERSION_MAJOR > (major)) ||                                \
     (PX_PHYSICS_VERSION_MAJOR == (major) &&                                \
      PX_PHYSICS_VERSION_MINOR >= (minor)))

#if KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 4)
#define KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
#endif

namespace KE::PhysXCompat {

inline void
setArticulationLimit(physx::PxArticulationJointReducedCoordinate& joint,
                     physx::PxArticulationAxis::Enum axis, float low,
                     float high) {
#if PX_PHYSICS_VERSION_MAJOR == 5 && PX_PHYSICS_VERSION_MINOR < 2
    joint.setLimit(axis, low, high);
#else
    joint.setLimitParams(axis, physx::PxArticulationLimit(low, high));
#endif
}

inline void setArticulationDrive(
    physx::PxArticulationJointReducedCoordinate& joint,
    physx::PxArticulationAxis::Enum axis, float stiffness, float damping,
    float maxForce,
    physx::PxArticulationDriveType::Enum driveType =
        physx::PxArticulationDriveType::eFORCE) {
#if PX_PHYSICS_VERSION_MAJOR == 5 && PX_PHYSICS_VERSION_MINOR < 2
    joint.setDrive(axis, stiffness, damping, maxForce, driveType);
#else
    joint.setDriveParams(
        axis,
        physx::PxArticulationDrive(stiffness, damping, maxForce, driveType));
#endif
}

} // namespace KE::PhysXCompat

#endif

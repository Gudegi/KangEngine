#ifndef _PHYSICS_GPU_SYSTEM_KERNELS_HPP_
#define _PHYSICS_GPU_SYSTEM_KERNELS_HPP_

#include "sim/gpu_array_view.hpp"

#include <cstddef>
#include <cstdint>

namespace KE {
namespace PhysicsGpuKernels {

void packSparseRigidCommandCUDA(const Sim::GpuArrayView& logicalIndices,
                                const void* physxGpuIndices,
                                const void* denseCommand,
                                void* packedPhysxGpuIndices,
                                void* packedCommand, uint32_t rigidCount,
                                uint64_t streamHandle);

void packSparseRigidStateCUDA(const Sim::GpuArrayView& logicalIndices,
                              const void* physxGpuIndices,
                              const void* denseRigidState,
                              void* packedPhysxGpuIndices, void* packedPose,
                              void* packedLinearVelocity,
                              void* packedAngularVelocity,
                              uint32_t rigidCount, uint64_t streamHandle,
                              size_t poseStride, size_t posePositionOffset,
                              size_t poseRotationOffset, size_t vec3Stride);

void packSparseArticulationDofCommandCUDA(
    const Sim::GpuArrayView& logicalIndices, const void* physxGpuIndices,
    const void* denseDofCommand, void* packedPhysxGpuIndices,
    void* packedDofCommand, uint32_t articulationCount, uint32_t maxDofs,
    uint64_t streamHandle);

void packSparseArticulationRootStateCUDA(
    const Sim::GpuArrayView& logicalIndices, const void* physxGpuIndices,
    const void* denseLinkState, void* packedPhysxGpuIndices, void* packedPose,
    void* packedLinearVelocity, void* packedAngularVelocity,
    uint32_t articulationCount, uint32_t maxLinks, uint64_t streamHandle,
    size_t poseStride, size_t posePositionOffset, size_t poseRotationOffset,
    size_t vec3Stride);

} // namespace PhysicsGpuKernels
} // namespace KE

#endif

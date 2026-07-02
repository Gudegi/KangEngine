#include "physics/physics_gpu_system_kernels.hpp"

#include <PxContact.h>
#include <cuda_runtime.h>

#include <initializer_list>
#include <stdexcept>
#include <string>

namespace KE {
namespace PhysicsGpuKernels {
namespace {

enum ContactSensorDescriptorField : uint32_t {
    SensorBodyKind = 0,
    SensorRowMapOffset,
    SensorRowMapCount,
    SensorBodyMapOffset,
    SensorBodyMapCount,
    SensorOutputOffset,
    SensorEnvironmentCount,
    SensorBodyCount,
    SensorDescriptorSize,
};

void checkCUDA(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

template <typename IndexT>
__global__ void packSparseRigidCommandKernel(
    const IndexT* logicalIndices, const uint32_t* physxGpuIndices,
    const float* denseCommand, uint32_t* packedPhysxGpuIndices,
    float* packedCommand, uint32_t count, uint32_t rigidCount) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    const int logical = static_cast<int>(logicalIndices[i]);
    if (logical < 0 || static_cast<uint32_t>(logical) >= rigidCount) {
        packedPhysxGpuIndices[i] = 0xFFFFFFFFu;
        packedCommand[i * 3 + 0] = 0.0f;
        packedCommand[i * 3 + 1] = 0.0f;
        packedCommand[i * 3 + 2] = 0.0f;
        return;
    }

    const uint32_t source = static_cast<uint32_t>(logical);
    packedPhysxGpuIndices[i] = physxGpuIndices[source];
    packedCommand[i * 3 + 0] = denseCommand[source * 3 + 0];
    packedCommand[i * 3 + 1] = denseCommand[source * 3 + 1];
    packedCommand[i * 3 + 2] = denseCommand[source * 3 + 2];
}

template <typename IndexT>
__global__ void packSparseRigidStateKernel(
    const IndexT* logicalIndices, const uint32_t* physxGpuIndices,
    const float* denseRigidState, uint32_t* packedPhysxGpuIndices,
    unsigned char* packedPose, unsigned char* packedLinearVelocity,
    unsigned char* packedAngularVelocity, uint32_t count, uint32_t rigidCount,
    size_t poseStride, size_t posePositionOffset, size_t poseRotationOffset,
    size_t vec3Stride) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    const int logical = static_cast<int>(logicalIndices[i]);
    if (logical < 0 || static_cast<uint32_t>(logical) >= rigidCount) {
        packedPhysxGpuIndices[i] = 0xFFFFFFFFu;
        return;
    }

    const uint32_t source = static_cast<uint32_t>(logical);
    const float* row = denseRigidState + static_cast<size_t>(source) * 13;
    packedPhysxGpuIndices[i] = physxGpuIndices[source];

    float* position = reinterpret_cast<float*>(
        packedPose + static_cast<size_t>(i) * poseStride + posePositionOffset);
    float* rotation = reinterpret_cast<float*>(
        packedPose + static_cast<size_t>(i) * poseStride + poseRotationOffset);
    float* linearVelocity = reinterpret_cast<float*>(
        packedLinearVelocity + static_cast<size_t>(i) * vec3Stride);
    float* angularVelocity = reinterpret_cast<float*>(
        packedAngularVelocity + static_cast<size_t>(i) * vec3Stride);

    position[0] = row[0];
    position[1] = row[1];
    position[2] = row[2];
    rotation[0] = row[3];
    rotation[1] = row[4];
    rotation[2] = row[5];
    rotation[3] = row[6];
    linearVelocity[0] = row[7];
    linearVelocity[1] = row[8];
    linearVelocity[2] = row[9];
    angularVelocity[0] = row[10];
    angularVelocity[1] = row[11];
    angularVelocity[2] = row[12];
}

template <typename IndexT>
__global__ void packSparseArticulationDofCommandKernel(
    const IndexT* logicalIndices, const uint32_t* physxGpuIndices,
    const float* denseDofCommand, uint32_t* packedPhysxGpuIndices,
    float* packedDofCommand, uint32_t count, uint32_t articulationCount,
    uint32_t maxDofs) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = count * maxDofs;
    if (item >= total)
        return;

    const uint32_t packedRow = item / maxDofs;
    const uint32_t dof = item - packedRow * maxDofs;
    const int logical = static_cast<int>(logicalIndices[packedRow]);
    if (logical < 0 || static_cast<uint32_t>(logical) >= articulationCount) {
        if (dof == 0)
            packedPhysxGpuIndices[packedRow] = 0xFFFFFFFFu;
        packedDofCommand[item] = 0.0f;
        return;
    }

    const uint32_t source = static_cast<uint32_t>(logical);
    if (dof == 0)
        packedPhysxGpuIndices[packedRow] = physxGpuIndices[source];
    packedDofCommand[item] =
        denseDofCommand[static_cast<size_t>(source) * maxDofs + dof];
}

template <typename IndexT>
__global__ void packSparseArticulationRootStateKernel(
    const IndexT* logicalIndices, const uint32_t* physxGpuIndices,
    const float* denseLinkState, uint32_t* packedPhysxGpuIndices,
    unsigned char* packedPose, unsigned char* packedLinearVelocity,
    unsigned char* packedAngularVelocity, uint32_t count,
    uint32_t articulationCount, uint32_t maxLinks, size_t poseStride,
    size_t posePositionOffset, size_t poseRotationOffset, size_t vec3Stride) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    const int logical = static_cast<int>(logicalIndices[i]);
    if (logical < 0 || static_cast<uint32_t>(logical) >= articulationCount) {
        packedPhysxGpuIndices[i] = 0xFFFFFFFFu;
        return;
    }

    const uint32_t source = static_cast<uint32_t>(logical);
    const float* row =
        denseLinkState + static_cast<size_t>(source) * maxLinks * 13;
    packedPhysxGpuIndices[i] = physxGpuIndices[source];

    float* position = reinterpret_cast<float*>(
        packedPose + static_cast<size_t>(i) * poseStride + posePositionOffset);
    float* rotation = reinterpret_cast<float*>(
        packedPose + static_cast<size_t>(i) * poseStride + poseRotationOffset);
    float* linearVelocity = reinterpret_cast<float*>(
        packedLinearVelocity + static_cast<size_t>(i) * vec3Stride);
    float* angularVelocity = reinterpret_cast<float*>(
        packedAngularVelocity + static_cast<size_t>(i) * vec3Stride);

    position[0] = row[0];
    position[1] = row[1];
    position[2] = row[2];
    rotation[0] = row[3];
    rotation[1] = row[4];
    rotation[2] = row[5];
    rotation[3] = row[6];
    linearVelocity[0] = row[7];
    linearVelocity[1] = row[8];
    linearVelocity[2] = row[9];
    angularVelocity[0] = row[10];
    angularVelocity[1] = row[11];
    angularVelocity[2] = row[12];
}

template <typename IndexT>
__global__ void clearSparseRigidCommandsKernel(const IndexT* logicalIndices,
                                               float* denseForce,
                                               float* denseTorque,
                                               uint32_t count,
                                               uint32_t rigidCount) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = count * 3;
    if (item >= total)
        return;

    const uint32_t packedRow = item / 3;
    const uint32_t axis = item - packedRow * 3;
    const int logical = static_cast<int>(logicalIndices[packedRow]);
    if (logical < 0 || static_cast<uint32_t>(logical) >= rigidCount)
        return;

    const size_t offset = static_cast<size_t>(logical) * 3 + axis;
    denseForce[offset] = 0.0f;
    denseTorque[offset] = 0.0f;
}

template <typename IndexT>
__global__ void clearSparseArticulationCommandsKernel(
    const IndexT* logicalIndices, const float* jointPositions,
    float* jointForces, float* targetJointPositions,
    float* targetJointVelocities, uint32_t count, uint32_t articulationCount,
    uint32_t maxDofs) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = count * maxDofs;
    if (item >= total)
        return;

    const uint32_t packedRow = item / maxDofs;
    const uint32_t dof = item - packedRow * maxDofs;
    const int logical = static_cast<int>(logicalIndices[packedRow]);
    if (logical < 0 || static_cast<uint32_t>(logical) >= articulationCount)
        return;

    const size_t offset = static_cast<size_t>(logical) * maxDofs + dof;
    targetJointPositions[offset] = jointPositions[offset];
    targetJointVelocities[offset] = 0.0f;
    jointForces[offset] = 0.0f;
}

__device__ void writeContactBodyRef(physx::PxNodeIndex node,
                                    const int32_t* nodeBodyRefs,
                                    uint32_t nodeBodyRefCapacity,
                                    int32_t* output) {
    output[0] = -1;
    output[1] = -1;
    output[2] = -1;
    if (!nodeBodyRefs || !node.isValid())
        return;
    const uint32_t nodeIndex = node.index();
    if (nodeIndex >= nodeBodyRefCapacity)
        return;
    const int32_t* source = nodeBodyRefs + static_cast<size_t>(nodeIndex) * 3;
    output[0] = source[0];
    output[1] = source[1];
    output[2] = source[2];
}

__global__ void flattenContactPairsKernel(const physx::PxGpuContactPair* pairs,
                                          const uint32_t* pairCount,
                                          uint64_t* contactPairHeaders,
                                          const int32_t* nodeBodyRefs,
                                          uint32_t nodeBodyRefCapacity,
                                          int32_t* contactPairBodyRefs,
                                          float* contactPoints,
                                          uint32_t* contactPointCount,
                                          uint32_t* contactPointPairIndices,
                                          uint32_t maxPairs,
                                          uint32_t maxContactPoints) {
    const uint32_t pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = pairCount ? min(*pairCount, maxPairs) : 0;
    if (pairIndex >= count)
        return;

    const physx::PxGpuContactPair pair = pairs[pairIndex];
    if (contactPairHeaders) {
        uint64_t* row = contactPairHeaders + static_cast<size_t>(pairIndex) * 6;
        row[0] = static_cast<uint64_t>(pair.nodeIndex0.getInd());
        row[1] = static_cast<uint64_t>(pair.nodeIndex1.getInd());
        row[2] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pair.actor0));
        row[3] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pair.actor1));
        row[4] = static_cast<uint64_t>(pair.transformCacheRef0);
        row[5] = static_cast<uint64_t>(pair.transformCacheRef1);
    }
    if (contactPairBodyRefs) {
        int32_t* row =
            contactPairBodyRefs + static_cast<size_t>(pairIndex) * 6;
        writeContactBodyRef(pair.nodeIndex0, nodeBodyRefs,
                            nodeBodyRefCapacity, row);
        writeContactBodyRef(pair.nodeIndex1, nodeBodyRefs,
                            nodeBodyRefCapacity, row + 3);
    }
    if (pair.nbContacts == 0 || pair.nbPatches == 0 || !pair.contactPatches ||
        !pair.contactPoints)
        return;

    physx::PxContactStreamIterator iterator(
        pair.contactPatches, pair.contactPoints, nullptr, pair.nbPatches,
        pair.nbContacts);
    uint32_t localContact = 0;
    while (iterator.hasNextPatch()) {
        iterator.nextPatch();
        while (iterator.hasNextContact()) {
            iterator.nextContact();
            const uint32_t outIndex = atomicAdd(contactPointCount, 1u);
            if (outIndex < maxContactPoints) {
                const physx::PxVec3& point = iterator.getContactPoint();
                const physx::PxVec3& normal = iterator.getContactNormal();
                const float force =
                    pair.contactForces ? pair.contactForces[localContact] : 0.0f;
                float* row = contactPoints + static_cast<size_t>(outIndex) * 10;
                if (contactPointPairIndices)
                    contactPointPairIndices[outIndex] = pairIndex;
                row[0] = point.x;
                row[1] = point.y;
                row[2] = point.z;
                row[3] = normal.x;
                row[4] = normal.y;
                row[5] = normal.z;
                row[6] = normal.x * force;
                row[7] = normal.y * force;
                row[8] = normal.z * force;
                row[9] = iterator.getSeparation();
            }
            ++localContact;
        }
    }
}

__device__ int32_t contactSensorOutput(
    const int32_t* ref, const int32_t* desc,
    const int32_t* rowToEnvironment, const int32_t* bodyToSlot,
    uint32_t outputCount) {
    const int32_t row = ref[1];
    const int32_t body = ref[2];
    if (ref[0] != desc[SensorBodyKind] || row < 0 || body < 0 ||
        row >= desc[SensorRowMapCount] || body >= desc[SensorBodyMapCount])
        return -1;

    const int32_t environment =
        rowToEnvironment[desc[SensorRowMapOffset] + row];
    const int32_t bodySlot = bodyToSlot[desc[SensorBodyMapOffset] + body];
    if (environment < 0 || bodySlot < 0 ||
        environment >= desc[SensorEnvironmentCount] ||
        bodySlot >= desc[SensorBodyCount])
        return -1;

    const int32_t output = desc[SensorOutputOffset] +
                           environment * desc[SensorBodyCount] + bodySlot;
    return output >= 0 && static_cast<uint32_t>(output) < outputCount
               ? output
               : -1;
}

__global__ void aggregateContactSensorsKernel(
    const int32_t* pairBodyRefs, const uint32_t* pairCount,
    const int32_t* sensorDescriptors, uint32_t sensorCount,
    const int32_t* rowToEnvironment, const int32_t* bodyToSlot,
    int32_t* contactCount, uint32_t outputCount, uint32_t maxPairs) {
    const uint32_t pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = pairCount ? min(*pairCount, maxPairs) : 0;
    if (pairIndex >= count)
        return;

    const int32_t* pair = pairBodyRefs + static_cast<size_t>(pairIndex) * 6;
    for (uint32_t endpoint = 0; endpoint < 2; ++endpoint) {
        const int32_t* ref = pair + endpoint * 3;
        for (uint32_t sensorIndex = 0; sensorIndex < sensorCount;
             ++sensorIndex) {
            const int32_t* desc =
                sensorDescriptors + sensorIndex * SensorDescriptorSize;
            const int32_t output = contactSensorOutput(
                ref, desc, rowToEnvironment, bodyToSlot, outputCount);
            if (output >= 0)
                atomicAdd(contactCount + output, 1);
        }
    }
}

__global__ void aggregateContactSensorImpulseKernel(
    const int32_t* pairBodyRefs, const uint32_t* pairCount,
    const float* contactPoints, const uint32_t* pointCount,
    const uint32_t* pointPairIndices, const int32_t* sensorDescriptors,
    uint32_t sensorCount, const int32_t* rowToEnvironment,
    const int32_t* bodyToSlot, float* netImpulse, uint32_t outputCount,
    uint32_t maxPairs, uint32_t maxPoints) {
    const uint32_t pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = pointCount ? min(*pointCount, maxPoints) : 0;
    if (pointIndex >= count)
        return;
    const uint32_t pairIndex = pointPairIndices[pointIndex];
    const uint32_t pairs = pairCount ? min(*pairCount, maxPairs) : 0;
    if (pairIndex >= pairs)
        return;

    const int32_t* pair = pairBodyRefs + static_cast<size_t>(pairIndex) * 6;
    const float* impulse = contactPoints + static_cast<size_t>(pointIndex) * 10 + 6;
    for (uint32_t endpoint = 0; endpoint < 2; ++endpoint) {
        const int32_t* ref = pair + endpoint * 3;
        const float sign = endpoint == 0 ? 1.0f : -1.0f;
        for (uint32_t sensorIndex = 0; sensorIndex < sensorCount;
             ++sensorIndex) {
            const int32_t* desc =
                sensorDescriptors + sensorIndex * SensorDescriptorSize;
            const int32_t output = contactSensorOutput(
                ref, desc, rowToEnvironment, bodyToSlot, outputCount);
            if (output < 0)
                continue;
            float* target = netImpulse + static_cast<size_t>(output) * 3;
            atomicAdd(target + 0, sign * impulse[0]);
            atomicAdd(target + 1, sign * impulse[1]);
            atomicAdd(target + 2, sign * impulse[2]);
        }
    }
}

__global__ void contactCountToMaskKernel(const int32_t* contactCount,
                                         uint8_t* inContact,
                                         uint32_t outputCount) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < outputCount)
        inContact[index] = contactCount[index] > 0;
}

uint32_t viewCount(const Sim::GpuArrayView& view) {
    const int64_t count = view.numel();
    if (count < 0)
        throw std::runtime_error("sparse rigid command indices are invalid");
    if (count > static_cast<int64_t>(UINT32_MAX))
        throw std::runtime_error(
            "sparse rigid command index count is too large");
    return static_cast<uint32_t>(count);
}

void validateSparseIndexView(const Sim::GpuArrayView& view, uint32_t maxCount,
                             const char* label) {
    if (!view.isCUDA())
        throw std::runtime_error(std::string(label) +
                                 " indices must be a CUDADevice view");
    if (view.dtype != Sim::SimDType::Int32 &&
        view.dtype != Sim::SimDType::UInt32)
        throw std::runtime_error(std::string(label) +
                                 " indices must be int32 or uint32");
    if (!view.data)
        throw std::runtime_error(std::string(label) +
                                 " indices require a non-null data pointer");
    if (view.shape.size() != 1)
        throw std::runtime_error(std::string(label) +
                                 " indices must have shape [count]");
    if (!view.strides.empty() &&
        (view.strides.size() != 1 || view.strides[0] != 1))
        throw std::runtime_error(std::string(label) +
                                 " indices must be contiguous");
    if (viewCount(view) > maxCount)
        throw std::runtime_error(std::string(label) +
                                 " index count exceeds source count");
}

void validateContactSensorView(const Sim::GpuArrayView& view,
                               Sim::SimDType dtype, size_t rank,
                               const char* label) {
    if (!view.isCUDA() || view.dtype != dtype || !view.data)
        throw std::runtime_error(std::string(label) +
                                 " must be a non-null CUDA view");
    if (view.shape.size() != rank)
        throw std::runtime_error(std::string(label) + " has invalid rank");
}

} // namespace

void packSparseRigidCommandCUDA(const Sim::GpuArrayView& logicalIndices,
                                const void* physxGpuIndices,
                                const void* denseCommand,
                                void* packedPhysxGpuIndices,
                                void* packedCommand, uint32_t rigidCount,
                                uint64_t streamHandle) {
    validateSparseIndexView(logicalIndices, rigidCount, "sparse rigid command");
    const uint32_t count = viewCount(logicalIndices);
    if (count == 0)
        return;

    if (logicalIndices.deviceId >= 0)
        checkCUDA(cudaSetDevice(logicalIndices.deviceId),
                  "cudaSetDevice(pack sparse rigid command)");

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (logicalIndices.readyEventHandle != 0) {
        auto event =
            reinterpret_cast<cudaEvent_t>(logicalIndices.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(pack sparse rigid command)");
    }

    const int blockSize = 256;
    const int gridSize = (static_cast<int>(count) + blockSize - 1) / blockSize;
    if (logicalIndices.dtype == Sim::SimDType::Int32) {
        packSparseRigidCommandKernel<<<gridSize, blockSize, 0, stream>>>(
            static_cast<const int32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseCommand),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<float*>(packedCommand), count, rigidCount);
    } else {
        packSparseRigidCommandKernel<<<gridSize, blockSize, 0, stream>>>(
            static_cast<const uint32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseCommand),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<float*>(packedCommand), count, rigidCount);
    }
    checkCUDA(cudaGetLastError(), "packSparseRigidCommandKernel");
}

void packSparseRigidStateCUDA(
    const Sim::GpuArrayView& logicalIndices, const void* physxGpuIndices,
    const void* denseRigidState, void* packedPhysxGpuIndices, void* packedPose,
    void* packedLinearVelocity, void* packedAngularVelocity,
    uint32_t rigidCount, uint64_t streamHandle, size_t poseStride,
    size_t posePositionOffset, size_t poseRotationOffset, size_t vec3Stride) {
    validateSparseIndexView(logicalIndices, rigidCount, "sparse rigid state");
    const uint32_t count = viewCount(logicalIndices);
    if (count == 0)
        return;

    if (logicalIndices.deviceId >= 0)
        checkCUDA(cudaSetDevice(logicalIndices.deviceId),
                  "cudaSetDevice(pack sparse rigid state)");

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (logicalIndices.readyEventHandle != 0) {
        auto event =
            reinterpret_cast<cudaEvent_t>(logicalIndices.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(pack sparse rigid state)");
    }

    const int blockSize = 256;
    const int gridSize = (static_cast<int>(count) + blockSize - 1) / blockSize;
    if (logicalIndices.dtype == Sim::SimDType::Int32) {
        packSparseRigidStateKernel<<<gridSize, blockSize, 0, stream>>>(
            static_cast<const int32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseRigidState),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<unsigned char*>(packedPose),
            static_cast<unsigned char*>(packedLinearVelocity),
            static_cast<unsigned char*>(packedAngularVelocity), count,
            rigidCount, poseStride, posePositionOffset, poseRotationOffset,
            vec3Stride);
    } else {
        packSparseRigidStateKernel<<<gridSize, blockSize, 0, stream>>>(
            static_cast<const uint32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseRigidState),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<unsigned char*>(packedPose),
            static_cast<unsigned char*>(packedLinearVelocity),
            static_cast<unsigned char*>(packedAngularVelocity), count,
            rigidCount, poseStride, posePositionOffset, poseRotationOffset,
            vec3Stride);
    }
    checkCUDA(cudaGetLastError(), "packSparseRigidStateKernel");
}

void packSparseArticulationDofCommandCUDA(
    const Sim::GpuArrayView& logicalIndices, const void* physxGpuIndices,
    const void* denseDofCommand, void* packedPhysxGpuIndices,
    void* packedDofCommand, uint32_t articulationCount, uint32_t maxDofs,
    uint64_t streamHandle) {
    validateSparseIndexView(logicalIndices, articulationCount,
                            "sparse articulation DOF command");
    const uint32_t count = viewCount(logicalIndices);
    if (count == 0 || maxDofs == 0)
        return;

    if (logicalIndices.deviceId >= 0)
        checkCUDA(cudaSetDevice(logicalIndices.deviceId),
                  "cudaSetDevice(pack sparse articulation dof command)");

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (logicalIndices.readyEventHandle != 0) {
        auto event =
            reinterpret_cast<cudaEvent_t>(logicalIndices.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(pack sparse articulation dof command)");
    }

    const int blockSize = 256;
    const int gridSize =
        (static_cast<int>(count * maxDofs) + blockSize - 1) / blockSize;
    if (logicalIndices.dtype == Sim::SimDType::Int32) {
        packSparseArticulationDofCommandKernel<<<gridSize, blockSize, 0,
                                                 stream>>>(
            static_cast<const int32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseDofCommand),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<float*>(packedDofCommand), count, articulationCount,
            maxDofs);
    } else {
        packSparseArticulationDofCommandKernel<<<gridSize, blockSize, 0,
                                                 stream>>>(
            static_cast<const uint32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseDofCommand),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<float*>(packedDofCommand), count, articulationCount,
            maxDofs);
    }
    checkCUDA(cudaGetLastError(), "packSparseArticulationDofCommandKernel");
}

void packSparseArticulationRootStateCUDA(
    const Sim::GpuArrayView& logicalIndices, const void* physxGpuIndices,
    const void* denseLinkState, void* packedPhysxGpuIndices, void* packedPose,
    void* packedLinearVelocity, void* packedAngularVelocity,
    uint32_t articulationCount, uint32_t maxLinks, uint64_t streamHandle,
    size_t poseStride, size_t posePositionOffset, size_t poseRotationOffset,
    size_t vec3Stride) {
    validateSparseIndexView(logicalIndices, articulationCount,
                            "sparse articulation root state");
    const uint32_t count = viewCount(logicalIndices);
    if (count == 0)
        return;

    if (logicalIndices.deviceId >= 0)
        checkCUDA(cudaSetDevice(logicalIndices.deviceId),
                  "cudaSetDevice(pack sparse articulation root state)");

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (logicalIndices.readyEventHandle != 0) {
        auto event =
            reinterpret_cast<cudaEvent_t>(logicalIndices.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(pack sparse articulation root state)");
    }

    const int blockSize = 256;
    const int gridSize = (static_cast<int>(count) + blockSize - 1) / blockSize;
    if (logicalIndices.dtype == Sim::SimDType::Int32) {
        packSparseArticulationRootStateKernel<<<gridSize, blockSize, 0,
                                                stream>>>(
            static_cast<const int32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseLinkState),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<unsigned char*>(packedPose),
            static_cast<unsigned char*>(packedLinearVelocity),
            static_cast<unsigned char*>(packedAngularVelocity), count,
            articulationCount, maxLinks, poseStride, posePositionOffset,
            poseRotationOffset, vec3Stride);
    } else {
        packSparseArticulationRootStateKernel<<<gridSize, blockSize, 0,
                                                stream>>>(
            static_cast<const uint32_t*>(logicalIndices.data),
            static_cast<const uint32_t*>(physxGpuIndices),
            static_cast<const float*>(denseLinkState),
            static_cast<uint32_t*>(packedPhysxGpuIndices),
            static_cast<unsigned char*>(packedPose),
            static_cast<unsigned char*>(packedLinearVelocity),
            static_cast<unsigned char*>(packedAngularVelocity), count,
            articulationCount, maxLinks, poseStride, posePositionOffset,
            poseRotationOffset, vec3Stride);
    }
    checkCUDA(cudaGetLastError(), "packSparseArticulationRootStateKernel");
}

void clearSparseRigidCommandsCUDA(const Sim::GpuArrayView& logicalIndices,
                                  void* denseForce, void* denseTorque,
                                  uint32_t rigidCount,
                                  uint64_t streamHandle) {
    validateSparseIndexView(logicalIndices, rigidCount,
                            "sparse rigid command clear");
    const uint32_t count = viewCount(logicalIndices);
    if (count == 0 || !denseForce || !denseTorque)
        return;

    if (logicalIndices.deviceId >= 0)
        checkCUDA(cudaSetDevice(logicalIndices.deviceId),
                  "cudaSetDevice(clear sparse rigid commands)");

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (logicalIndices.readyEventHandle != 0) {
        auto event =
            reinterpret_cast<cudaEvent_t>(logicalIndices.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(clear sparse rigid commands)");
    }

    const int blockSize = 256;
    const int gridSize =
        (static_cast<int>(count * 3) + blockSize - 1) / blockSize;
    if (logicalIndices.dtype == Sim::SimDType::Int32) {
        clearSparseRigidCommandsKernel<<<gridSize, blockSize, 0, stream>>>(
            static_cast<const int32_t*>(logicalIndices.data),
            static_cast<float*>(denseForce), static_cast<float*>(denseTorque),
            count, rigidCount);
    } else {
        clearSparseRigidCommandsKernel<<<gridSize, blockSize, 0, stream>>>(
            static_cast<const uint32_t*>(logicalIndices.data),
            static_cast<float*>(denseForce), static_cast<float*>(denseTorque),
            count, rigidCount);
    }
    checkCUDA(cudaGetLastError(), "clearSparseRigidCommandsKernel");
}

void clearSparseArticulationCommandsCUDA(
    const Sim::GpuArrayView& logicalIndices, const void* jointPositions,
    void* jointForces, void* targetJointPositions, void* targetJointVelocities,
    uint32_t articulationCount, uint32_t maxDofs, uint64_t streamHandle) {
    validateSparseIndexView(logicalIndices, articulationCount,
                            "sparse articulation command clear");
    const uint32_t count = viewCount(logicalIndices);
    if (count == 0 || maxDofs == 0 || !jointPositions || !jointForces ||
        !targetJointPositions || !targetJointVelocities)
        return;

    if (logicalIndices.deviceId >= 0)
        checkCUDA(cudaSetDevice(logicalIndices.deviceId),
                  "cudaSetDevice(clear sparse articulation commands)");

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (logicalIndices.readyEventHandle != 0) {
        auto event =
            reinterpret_cast<cudaEvent_t>(logicalIndices.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(clear sparse articulation commands)");
    }

    const int blockSize = 256;
    const int gridSize =
        (static_cast<int>(count * maxDofs) + blockSize - 1) / blockSize;
    if (logicalIndices.dtype == Sim::SimDType::Int32) {
        clearSparseArticulationCommandsKernel<<<gridSize, blockSize, 0,
                                                stream>>>(
            static_cast<const int32_t*>(logicalIndices.data),
            static_cast<const float*>(jointPositions),
            static_cast<float*>(jointForces),
            static_cast<float*>(targetJointPositions),
            static_cast<float*>(targetJointVelocities), count,
            articulationCount, maxDofs);
    } else {
        clearSparseArticulationCommandsKernel<<<gridSize, blockSize, 0,
                                                stream>>>(
            static_cast<const uint32_t*>(logicalIndices.data),
            static_cast<const float*>(jointPositions),
            static_cast<float*>(jointForces),
            static_cast<float*>(targetJointPositions),
            static_cast<float*>(targetJointVelocities), count,
            articulationCount, maxDofs);
    }
    checkCUDA(cudaGetLastError(), "clearSparseArticulationCommandsKernel");
}

void flattenContactPairsCUDA(const void* contactPairs, const void* pairCount,
                             void* contactPairHeaders,
                             const void* contactNodeBodyRefs,
                             uint32_t contactNodeBodyRefCapacity,
                             void* contactPairBodyRefs, void* contactPoints,
                             void* contactPointCount,
                             void* contactPointPairIndices, uint32_t maxPairs,
                             uint32_t maxContactPoints,
                             uint64_t streamHandle) {
    if (!contactPairs || !pairCount || !contactPoints || !contactPointCount ||
        maxPairs == 0 || maxContactPoints == 0)
        return;

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (contactPairHeaders)
        checkCUDA(cudaMemsetAsync(contactPairHeaders, 0,
                                  sizeof(uint64_t) *
                                      static_cast<size_t>(maxPairs) * 6,
                                  stream),
                  "cudaMemsetAsync(contact pair headers)");
    if (contactPairBodyRefs)
        checkCUDA(cudaMemsetAsync(contactPairBodyRefs, 0xFF,
                                  sizeof(int32_t) *
                                      static_cast<size_t>(maxPairs) * 6,
                                  stream),
                  "cudaMemsetAsync(contact pair body refs)");
    checkCUDA(cudaMemsetAsync(contactPointCount, 0, sizeof(uint32_t), stream),
              "cudaMemsetAsync(contact point count)");
    if (contactPointPairIndices)
        checkCUDA(cudaMemsetAsync(contactPointPairIndices, 0xFF,
                                  sizeof(uint32_t) *
                                      static_cast<size_t>(maxContactPoints),
                                  stream),
                  "cudaMemsetAsync(contact point pair indices)");

    const int blockSize = 128;
    const int gridSize = (static_cast<int>(maxPairs) + blockSize - 1) / blockSize;
    flattenContactPairsKernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const physx::PxGpuContactPair*>(contactPairs),
        static_cast<const uint32_t*>(pairCount),
        static_cast<uint64_t*>(contactPairHeaders),
        static_cast<const int32_t*>(contactNodeBodyRefs),
        contactNodeBodyRefCapacity,
        static_cast<int32_t*>(contactPairBodyRefs),
        static_cast<float*>(contactPoints),
        static_cast<uint32_t*>(contactPointCount),
        static_cast<uint32_t*>(contactPointPairIndices), maxPairs,
        maxContactPoints);
    checkCUDA(cudaGetLastError(), "flattenContactPairsKernel");
}

void aggregateContactSensorsCUDA(
    const Sim::GpuArrayView& contactPairBodyRefs,
    const Sim::GpuArrayView& contactPairCount,
    const Sim::GpuArrayView& contactPoints,
    const Sim::GpuArrayView& contactPointCount,
    const Sim::GpuArrayView& contactPointPairIndices,
    const Sim::GpuArrayView& sensorDescriptors,
    const Sim::GpuArrayView& rowToEnvironment,
    const Sim::GpuArrayView& bodyToSlot,
    Sim::GpuArrayView& contactCount, Sim::GpuArrayView& inContact,
    Sim::GpuArrayView& netImpulse) {
    validateContactSensorView(contactPairBodyRefs, Sim::SimDType::Int32, 2,
                              "contact pair body refs");
    validateContactSensorView(contactPairCount, Sim::SimDType::UInt32, 1,
                              "contact pair count");
    validateContactSensorView(contactPoints, Sim::SimDType::Float32, 2,
                              "contact points");
    validateContactSensorView(contactPointCount, Sim::SimDType::UInt32, 1,
                              "contact point count");
    validateContactSensorView(contactPointPairIndices,
                              Sim::SimDType::UInt32, 1,
                              "contact point pair indices");
    validateContactSensorView(sensorDescriptors, Sim::SimDType::Int32, 2,
                              "contact sensor descriptors");
    validateContactSensorView(rowToEnvironment, Sim::SimDType::Int32, 1,
                              "contact sensor row map");
    validateContactSensorView(bodyToSlot, Sim::SimDType::Int32, 1,
                              "contact sensor body map");
    validateContactSensorView(contactCount, Sim::SimDType::Int32, 1,
                              "contact sensor count output");
    validateContactSensorView(inContact, Sim::SimDType::Bool, 1,
                              "contact sensor mask output");
    validateContactSensorView(netImpulse, Sim::SimDType::Float32, 2,
                              "contact sensor impulse output");
    if (contactPairBodyRefs.shape[1] != 6 || contactPairCount.shape[0] < 1 ||
        contactPoints.shape[1] != 10 || contactPointCount.shape[0] < 1 ||
        contactPointPairIndices.shape[0] != contactPoints.shape[0] ||
        sensorDescriptors.shape[1] != SensorDescriptorSize)
        throw std::runtime_error("contact sensor input shape mismatch");
    if (contactCount.shape != inContact.shape || contactCount.shape[0] <= 0)
        throw std::runtime_error("contact sensor output shape mismatch");
    if (netImpulse.shape[0] != contactCount.shape[0] ||
        netImpulse.shape[1] != 3)
        throw std::runtime_error("contact sensor impulse shape mismatch");
    if (contactCount.streamHandle != inContact.streamHandle ||
        contactCount.streamHandle != netImpulse.streamHandle)
        throw std::runtime_error(
            "contact sensor outputs must use the same CUDA stream");

    if (contactCount.deviceId >= 0)
        checkCUDA(cudaSetDevice(contactCount.deviceId),
                  "cudaSetDevice(aggregate contact sensor)");
    auto stream = reinterpret_cast<cudaStream_t>(contactCount.streamHandle);
    for (uint64_t eventHandle : {contactPairBodyRefs.readyEventHandle,
                                 contactPairCount.readyEventHandle}) {
        if (eventHandle == 0)
            continue;
        auto event = reinterpret_cast<cudaEvent_t>(eventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(aggregate contact sensor)");
    }

    checkCUDA(cudaMemsetAsync(contactCount.data, 0, contactCount.byteSize(),
                              stream),
              "cudaMemsetAsync(contact sensor count)");
    checkCUDA(cudaMemsetAsync(netImpulse.data, 0, netImpulse.byteSize(), stream),
              "cudaMemsetAsync(contact sensor impulse)");
    const uint32_t maxPairs =
        static_cast<uint32_t>(contactPairBodyRefs.shape[0]);
    const uint32_t outputCount = static_cast<uint32_t>(contactCount.numel());
    const int blockSize = 128;
    const int gridSize =
        (static_cast<int>(maxPairs) + blockSize - 1) / blockSize;
    aggregateContactSensorsKernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const int32_t*>(contactPairBodyRefs.data),
        static_cast<const uint32_t*>(contactPairCount.data),
        static_cast<const int32_t*>(sensorDescriptors.data),
        static_cast<uint32_t>(sensorDescriptors.shape[0]),
        static_cast<const int32_t*>(rowToEnvironment.data),
        static_cast<const int32_t*>(bodyToSlot.data),
        static_cast<int32_t*>(contactCount.data),
        outputCount, maxPairs);
    checkCUDA(cudaGetLastError(), "aggregateContactSensorsKernel");
    const uint32_t maxPoints = static_cast<uint32_t>(contactPoints.shape[0]);
    const int pointGridSize =
        (static_cast<int>(maxPoints) + blockSize - 1) / blockSize;
    aggregateContactSensorImpulseKernel<<<pointGridSize, blockSize, 0, stream>>>(
        static_cast<const int32_t*>(contactPairBodyRefs.data),
        static_cast<const uint32_t*>(contactPairCount.data),
        static_cast<const float*>(contactPoints.data),
        static_cast<const uint32_t*>(contactPointCount.data),
        static_cast<const uint32_t*>(contactPointPairIndices.data),
        static_cast<const int32_t*>(sensorDescriptors.data),
        static_cast<uint32_t>(sensorDescriptors.shape[0]),
        static_cast<const int32_t*>(rowToEnvironment.data),
        static_cast<const int32_t*>(bodyToSlot.data),
        static_cast<float*>(netImpulse.data), outputCount, maxPairs, maxPoints);
    checkCUDA(cudaGetLastError(), "aggregateContactSensorImpulseKernel");
    const int maskGridSize =
        (static_cast<int>(outputCount) + blockSize - 1) / blockSize;
    contactCountToMaskKernel<<<maskGridSize, blockSize, 0, stream>>>(
        static_cast<const int32_t*>(contactCount.data),
        static_cast<uint8_t*>(inContact.data), outputCount);
    checkCUDA(cudaGetLastError(), "contactCountToMaskKernel");
}

} // namespace PhysicsGpuKernels
} // namespace KE

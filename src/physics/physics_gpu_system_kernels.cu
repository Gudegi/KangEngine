#include "physics/physics_gpu_system_kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace KE {
namespace PhysicsGpuKernels {
namespace {

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

} // namespace PhysicsGpuKernels
} // namespace KE

#include "sim/gpu_transform_kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace KE {
namespace Sim {
namespace {

void checkCUDA(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

__device__ void quatToMat3(float x, float y, float z, float w, float* m) {
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;

    m[0] = 1.0f - 2.0f * (yy + zz);
    m[1] = 2.0f * (xy + wz);
    m[2] = 2.0f * (xz - wy);

    m[4] = 2.0f * (xy - wz);
    m[5] = 1.0f - 2.0f * (xx + zz);
    m[6] = 2.0f * (yz + wx);

    m[8] = 2.0f * (xz + wy);
    m[9] = 2.0f * (yz - wx);
    m[10] = 1.0f - 2.0f * (xx + yy);
}

__global__ void rigidStateToMat4Kernel(const float* rigidState, float* out,
                                       int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    const float* row = rigidState + static_cast<size_t>(i) * 13;
    float* m = out + static_cast<size_t>(i) * 16;

    quatToMat3(row[3], row[4], row[5], row[6], m);
    m[3] = 0.0f;
    m[7] = 0.0f;
    m[11] = 0.0f;
    m[12] = row[0];
    m[13] = row[1];
    m[14] = row[2];
    m[15] = 1.0f;
}

} // namespace


CUDAExternalTransformBuffer::CUDAExternalTransformBuffer(
    int count, int deviceId, std::string name) {
    allocate(count, deviceId, std::move(name));
}

CUDAExternalTransformBuffer::~CUDAExternalTransformBuffer() { release(); }

void CUDAExternalTransformBuffer::allocate(int count, int deviceId,
                                             std::string name) {
    if (count < 0)
        throw std::runtime_error(
            "CUDAExternalTransformBuffer count cannot be negative");
    release();
    if (count == 0)
        return;

    checkCUDA(cudaSetDevice(deviceId),
              "cudaSetDevice(CUDAExternalTransformBuffer)");
    void* data = nullptr;
    checkCUDA(cudaMalloc(&data, sizeof(float) * static_cast<size_t>(count) * 16),
              "cudaMalloc(CUDAExternalTransformBuffer)");

    _count = count;
    _view.data = data;
    _view.memoryType = SimMemoryType::CUDADevice;
    _view.dtype = SimDType::Float32;
    _view.lifetime = SimLifetimePolicy::ExternalOwner;
    _view.deviceId = deviceId;
    _view.shape = {count, 4, 4};
    _view.strides = {16, 4, 1};
    _view.name = std::move(name);
}

void CUDAExternalTransformBuffer::release() {
    if (_view.data)
        cudaFree(_view.data);
    _count = 0;
    _view = {};
}

uint64_t CUDAExternalTransformBuffer::incrementVersion() {
    return ++_view.version;
}

// Convert PhysX rigid state rows into renderer Mat4 transforms.
void launchRigidStateToMat4CUDA(const GpuArrayView& rigidState,
                                GpuArrayView& transforms, int count) {
    if (!rigidState.isCUDA() || !transforms.isCUDA())
        throw std::runtime_error(
            "launchRigidStateToMat4CUDA requires CUDADevice views");
    if (rigidState.dtype != SimDType::Float32 ||
        transforms.dtype != SimDType::Float32)
        throw std::runtime_error(
            "launchRigidStateToMat4CUDA requires float32 views");
    if (!rigidState.data || !transforms.data)
        throw std::runtime_error(
            "launchRigidStateToMat4CUDA requires non-null buffers");
    if (rigidState.shape.size() != 2 || rigidState.shape[1] != 13)
        throw std::runtime_error(
            "rigid state view must have shape [N, 13]");
    if (transforms.shape.size() != 3 || transforms.shape[1] != 4 ||
        transforms.shape[2] != 4)
        throw std::runtime_error(
            "transform view must have shape [N, 4, 4]");
    if (count < 0 || rigidState.shape[0] < count || transforms.shape[0] < count)
        throw std::runtime_error(
            "launchRigidStateToMat4CUDA count exceeds view shape");

    if (rigidState.deviceId >= 0)
        checkCUDA(cudaSetDevice(rigidState.deviceId), "cudaSetDevice");

    auto stream = reinterpret_cast<cudaStream_t>(rigidState.streamHandle);
    if (rigidState.readyEventHandle != 0) {
        auto event = reinterpret_cast<cudaEvent_t>(rigidState.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(rigid state to mat4)");
    }

    const int blockSize = 256;
    const int gridSize = (count + blockSize - 1) / blockSize;
    rigidStateToMat4Kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(rigidState.data),
        static_cast<float*>(transforms.data), count);
    checkCUDA(cudaGetLastError(), "rigidStateToMat4Kernel");
}

} // namespace Sim
} // namespace KE

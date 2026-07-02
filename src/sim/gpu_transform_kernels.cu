#include "sim/gpu_transform_kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace KE {
namespace Sim {
namespace {

constexpr int kMaxMappedArticulationLinks = 128;

struct MappedTransformPointers {
    float* links[kMaxMappedArticulationLinks];
};

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

__global__ void indexedRigidStateToMat4Kernel(const float* rigidState,
                                              const int* rigidRows,
                                              float* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    const int rowIndex = rigidRows[i];
    const float* row = rigidState + static_cast<size_t>(rowIndex) * 13;
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

__global__ void articulationLinkStateToMat4Kernel(
    const float* linkState, const int* articulationRows,
    const int* linkIndices, float* out, int envCount, int linkCount,
    int maxLinks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int transformCount = envCount * linkCount;
    if (index >= transformCount)
        return;

    const int linkIndex = index / envCount;
    const int envIndex = index % envCount;
    const int articulationRow = articulationRows[envIndex];
    const int stateLinkIndex = linkIndices[linkIndex];
    const size_t stateRow = static_cast<size_t>(articulationRow) * maxLinks +
                            stateLinkIndex;
    const float* row = linkState + stateRow * 13;
    float* m = out + static_cast<size_t>(index) * 16;

    quatToMat3(row[3], row[4], row[5], row[6], m);
    m[3] = 0.0f;
    m[7] = 0.0f;
    m[11] = 0.0f;
    m[12] = row[0];
    m[13] = row[1];
    m[14] = row[2];
    m[15] = 1.0f;
}

__global__ void articulationLinkStateToMappedMat4Kernel(
    const float* linkState, const int* articulationRows,
    const int* linkIndices, MappedTransformPointers outputs, int envCount,
    int linkCount, int maxLinks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int transformCount = envCount * linkCount;
    if (index >= transformCount)
        return;

    const int linkIndex = index / envCount;
    const int envIndex = index % envCount;
    const int articulationRow = articulationRows[envIndex];
    const int stateLinkIndex = linkIndices[linkIndex];
    const size_t stateRow = static_cast<size_t>(articulationRow) * maxLinks +
                            stateLinkIndex;
    const float* row = linkState + stateRow * 13;
    float* m = outputs.links[linkIndex] + static_cast<size_t>(envIndex) * 16;

    quatToMat3(row[3], row[4], row[5], row[6], m);
    m[3] = 0.0f;
    m[7] = 0.0f;
    m[11] = 0.0f;
    m[12] = row[0];
    m[13] = row[1];
    m[14] = row[2];
    m[15] = 1.0f;
}

void validateMat4Output(const GpuArrayView& transforms, int count) {
    const bool cudaAccessible =
        transforms.memoryType == SimMemoryType::CUDADevice ||
        transforms.memoryType == SimMemoryType::OpenGLBuffer;
    if (!cudaAccessible || transforms.dtype != SimDType::Float32 ||
        !transforms.data)
        throw std::runtime_error(
            "transform output must be a non-null CUDA float32 view");
    if (transforms.shape.size() != 3 || transforms.shape[0] < count ||
        transforms.shape[1] != 4 || transforms.shape[2] != 4)
        throw std::runtime_error(
            "transform output must have shape [N, 4, 4]");
}

void validateIndexView(const GpuArrayView& indices, int count,
                       int deviceId) {
    if (!indices.isCUDA() || indices.dtype != SimDType::Int32 ||
        !indices.data)
        throw std::runtime_error(
            "GPU transform indices must be a non-null CUDA int32 view");
    if (indices.shape.size() != 1 || indices.shape[0] != count)
        throw std::runtime_error(
            "GPU transform indices must have shape [N]");
    if (deviceId >= 0 && indices.deviceId >= 0 &&
        indices.deviceId != deviceId)
        throw std::runtime_error(
            "GPU transform indices and state must use the same CUDA device");
}

cudaStream_t transformStream(const GpuArrayView& state,
                             const GpuArrayView& transforms) {
    auto stream = reinterpret_cast<cudaStream_t>(transforms.streamHandle);
    if (state.readyEventHandle != 0) {
        auto event = reinterpret_cast<cudaEvent_t>(state.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(state to mat4)");
    }
    return stream;
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

void launchIndexedRigidStateToMat4CUDA(const GpuArrayView& rigidState,
                                       const GpuArrayView& rigidRows,
                                       GpuArrayView& transforms) {
    if (!rigidState.isCUDA() || rigidState.dtype != SimDType::Float32 ||
        !rigidState.data || rigidState.shape.size() != 2 ||
        rigidState.shape[1] != 13)
        throw std::runtime_error(
            "rigid state must be a non-null CUDA float32 [N, 13] view");
    const int count = static_cast<int>(rigidRows.shape.empty()
                                           ? 0
                                           : rigidRows.shape[0]);
    validateIndexView(rigidRows, count, rigidState.deviceId);
    validateMat4Output(transforms, count);
    if (rigidState.deviceId >= 0 && transforms.deviceId >= 0 &&
        rigidState.deviceId != transforms.deviceId)
        throw std::runtime_error(
            "rigid state and transform output must use the same CUDA device");
    if (count == 0)
        return;

    if (rigidState.deviceId >= 0)
        checkCUDA(cudaSetDevice(rigidState.deviceId), "cudaSetDevice");
    auto stream = transformStream(rigidState, transforms);
    const int blockSize = 256;
    const int gridSize = (count + blockSize - 1) / blockSize;
    indexedRigidStateToMat4Kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(rigidState.data),
        static_cast<const int*>(rigidRows.data),
        static_cast<float*>(transforms.data), count);
    checkCUDA(cudaGetLastError(), "indexedRigidStateToMat4Kernel");
}

void launchArticulationLinkStateToMat4CUDA(
    const GpuArrayView& articulationLinkState,
    const GpuArrayView& articulationRows, const GpuArrayView& linkIndices,
    GpuArrayView& transforms, int linkCount) {
    if (!articulationLinkState.isCUDA() ||
        articulationLinkState.dtype != SimDType::Float32 ||
        !articulationLinkState.data ||
        articulationLinkState.shape.size() != 3 ||
        articulationLinkState.shape[2] != 13)
        throw std::runtime_error(
            "articulation link state must be a non-null CUDA float32 "
            "[A, L, 13] view");
    const int envCount = static_cast<int>(articulationRows.shape.empty()
                                              ? 0
                                              : articulationRows.shape[0]);
    const int maxLinks =
        static_cast<int>(articulationLinkState.shape[1]);
    if (linkCount < 0 || linkCount > maxLinks)
        throw std::runtime_error(
            "articulation link transform count exceeds state max links");
    validateIndexView(articulationRows, envCount,
                      articulationLinkState.deviceId);
    validateIndexView(linkIndices, linkCount,
                      articulationLinkState.deviceId);
    if (!transforms.isCUDA() || transforms.dtype != SimDType::Float32 ||
        !transforms.data || transforms.shape.size() != 4 ||
        transforms.shape[0] < linkCount ||
        transforms.shape[1] != envCount || transforms.shape[2] != 4 ||
        transforms.shape[3] != 4)
        throw std::runtime_error(
            "articulation transform output must have shape [L, N, 4, 4]");
    if (articulationLinkState.deviceId >= 0 && transforms.deviceId >= 0 &&
        articulationLinkState.deviceId != transforms.deviceId)
        throw std::runtime_error(
            "articulation state and transforms must use the same CUDA device");
    if (envCount == 0 || linkCount == 0)
        return;

    if (articulationLinkState.deviceId >= 0)
        checkCUDA(cudaSetDevice(articulationLinkState.deviceId),
                  "cudaSetDevice");
    auto stream = transformStream(articulationLinkState, transforms);
    const int count = envCount * linkCount;
    const int blockSize = 256;
    const int gridSize = (count + blockSize - 1) / blockSize;
    articulationLinkStateToMat4Kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(articulationLinkState.data),
        static_cast<const int*>(articulationRows.data),
        static_cast<const int*>(linkIndices.data),
        static_cast<float*>(transforms.data), envCount, linkCount, maxLinks);
    checkCUDA(cudaGetLastError(), "articulationLinkStateToMat4Kernel");
}

void launchArticulationLinkStateToMappedMat4CUDA(
    const GpuArrayView& articulationLinkState,
    const GpuArrayView& articulationRows, const GpuArrayView& linkIndices,
    const std::vector<GpuArrayView>& mappedTransforms) {
    if (mappedTransforms.empty())
        return;
    if (mappedTransforms.size() > kMaxMappedArticulationLinks)
        throw std::runtime_error(
            "mapped articulation link count exceeds kernel limit");
    if (!articulationLinkState.isCUDA() ||
        articulationLinkState.dtype != SimDType::Float32 ||
        !articulationLinkState.data ||
        articulationLinkState.shape.size() != 3 ||
        articulationLinkState.shape[2] != 13)
        throw std::runtime_error(
            "articulation link state must be CUDA float32 [A, L, 13]");

    const int linkCount = static_cast<int>(mappedTransforms.size());
    const int envCount = static_cast<int>(articulationRows.shape[0]);
    const int maxLinks =
        static_cast<int>(articulationLinkState.shape[1]);
    validateIndexView(articulationRows, envCount,
                      articulationLinkState.deviceId);
    validateIndexView(linkIndices, linkCount,
                      articulationLinkState.deviceId);

    MappedTransformPointers outputs{};
    uint64_t streamHandle = mappedTransforms[0].streamHandle;
    for (int link = 0; link < linkCount; ++link) {
        const auto& output = mappedTransforms[static_cast<size_t>(link)];
        validateMat4Output(output, envCount);
        if (output.memoryType != SimMemoryType::OpenGLBuffer)
            throw std::runtime_error(
                "direct articulation output must be a mapped OpenGL buffer");
        if (output.streamHandle != streamHandle)
            throw std::runtime_error(
                "mapped articulation outputs must use one CUDA stream");
        outputs.links[link] = static_cast<float*>(output.data);
    }

    if (articulationLinkState.deviceId >= 0)
        checkCUDA(cudaSetDevice(articulationLinkState.deviceId),
                  "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    if (articulationLinkState.readyEventHandle != 0) {
        auto event = reinterpret_cast<cudaEvent_t>(
            articulationLinkState.readyEventHandle);
        checkCUDA(cudaStreamWaitEvent(stream, event, 0),
                  "cudaStreamWaitEvent(mapped articulation transforms)");
    }
    const int count = envCount * linkCount;
    const int blockSize = 256;
    const int gridSize = (count + blockSize - 1) / blockSize;
    articulationLinkStateToMappedMat4Kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(articulationLinkState.data),
        static_cast<const int*>(articulationRows.data),
        static_cast<const int*>(linkIndices.data), outputs, envCount, linkCount,
        maxLinks);
    checkCUDA(cudaGetLastError(),
              "articulationLinkStateToMappedMat4Kernel");
}

} // namespace Sim
} // namespace KE

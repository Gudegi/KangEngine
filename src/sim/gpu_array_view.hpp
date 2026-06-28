#ifndef _GPU_ARRAY_VIEW_HPP_
#define _GPU_ARRAY_VIEW_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace KE {
namespace Sim {

enum class SimMemoryType {
    CpuHost,
    CpuPinned,
    CudaDevice,
    OpenGLBuffer,
    VulkanBuffer,
    WebGPUBuffer,
    External,
};

enum class SimDType {
    Float32,
    Float64,
    Int32,
    UInt32,
    Int64,
    UInt64,
    UInt8,
    Bool,
    Unknown,
};

enum class SimLifetimePolicy {
    Borrowed,
    SharedOwner,
    ExternalOwner,
};

inline size_t simDTypeSize(SimDType dtype) {
    switch (dtype) {
    case SimDType::Float32:
    case SimDType::Int32:
    case SimDType::UInt32:
        return 4;
    case SimDType::Float64:
    case SimDType::Int64:
    case SimDType::UInt64:
        return 8;
    case SimDType::UInt8:
    case SimDType::Bool:
        return 1;
    case SimDType::Unknown:
        return 0;
    }
    return 0;
}

// Backend-neutral descriptor for CPU/GPU simulation buffers.
//
// This is a metadata view, not an owner. The producer is responsible for memory
// lifetime and synchronization. CUDA/WebGPU/Vulkan implementations should build
// on this shape instead of exposing raw pointers as the public API.
struct GpuArrayView {
    void* data = nullptr;
    SimMemoryType memoryType = SimMemoryType::CpuHost;
    SimDType dtype = SimDType::Unknown;
    SimLifetimePolicy lifetime = SimLifetimePolicy::Borrowed;
    int deviceId = -1;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides; // element strides, not byte strides
    uint64_t version = 0;
    uint64_t streamHandle = 0;
    uint64_t readyEventHandle = 0;
    std::shared_ptr<void> owner;
    std::string name;

    bool empty() const { return data == nullptr || numel() == 0; }
    bool hasOwner() const { return owner != nullptr; }
    bool isCuda() const { return memoryType == SimMemoryType::CudaDevice; }
    bool isCpu() const {
        return memoryType == SimMemoryType::CpuHost ||
               memoryType == SimMemoryType::CpuPinned;
    }

    int64_t numel() const {
        if (shape.empty())
            return 0;
        int64_t count = 1;
        for (int64_t extent : shape)
            count *= extent;
        return count;
    }

    size_t byteSize() const {
        return static_cast<size_t>(numel()) * simDTypeSize(dtype);
    }
};

} // namespace Sim
} // namespace KE

#endif

#include "sim/gpu_transform_kernels.hpp"

#include <stdexcept>

namespace KE {
namespace Sim {

CUDAExternalTransformBuffer::CUDAExternalTransformBuffer(
    int count, int deviceId, std::string name) {
    allocate(count, deviceId, name);
}

CUDAExternalTransformBuffer::~CUDAExternalTransformBuffer() { release(); }

void CUDAExternalTransformBuffer::allocate(int, int, std::string) {
    throw std::runtime_error(
        "CUDAExternalTransformBuffer requires KANGENGINE_USE_CUDA");
}

void CUDAExternalTransformBuffer::release() {
    _count = 0;
    _view = {};
}

uint64_t CUDAExternalTransformBuffer::incrementVersion() {
    return ++_view.version;
}

void launchRigidStateToMat4CUDA(const GpuArrayView&, GpuArrayView&, int) {
    throw std::runtime_error(
        "launchRigidStateToMat4CUDA requires KANGENGINE_USE_CUDA");
}

} // namespace Sim
} // namespace KE

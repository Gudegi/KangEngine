#include "sim/gpu_transform_kernels.hpp"

#include <stdexcept>

namespace KE {
namespace Sim {

void launchRigidStateToMat4CUDA(const GpuArrayView&, GpuArrayView&, int) {
    throw std::runtime_error(
        "launchRigidStateToMat4CUDA requires KANGENGINE_USE_CUDA");
}

} // namespace Sim
} // namespace KE

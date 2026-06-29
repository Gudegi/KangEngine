#ifndef _GPU_TRANSFORM_KERNELS_HPP_
#define _GPU_TRANSFORM_KERNELS_HPP_

#include "sim/gpu_array_view.hpp"

#include <cstdint>

namespace KE {
namespace Sim {

void launchRigidStateToMat4CUDA(const GpuArrayView& rigidState,
                                GpuArrayView& transforms, int count);

} // namespace Sim
} // namespace KE

#endif

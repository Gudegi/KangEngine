#ifndef _GPU_TRANSFORM_KERNELS_HPP_
#define _GPU_TRANSFORM_KERNELS_HPP_

#include "sim/gpu_array_view.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace KE {
namespace Sim {

class CUDAExternalTransformBuffer {
  public:
    CUDAExternalTransformBuffer() = default;
    CUDAExternalTransformBuffer(int count, int deviceId, std::string name = {});
    ~CUDAExternalTransformBuffer();

    CUDAExternalTransformBuffer(const CUDAExternalTransformBuffer&) = delete;
    CUDAExternalTransformBuffer& operator=(const CUDAExternalTransformBuffer&) = delete;
    CUDAExternalTransformBuffer(CUDAExternalTransformBuffer&&) = delete;
    CUDAExternalTransformBuffer& operator=(CUDAExternalTransformBuffer&&) = delete;

    void allocate(int count, int deviceId, std::string name = {});
    void release();
    uint64_t incrementVersion();

    bool empty() const { return _view.empty(); }
    int count() const { return _count; }
    GpuArrayView& view() { return _view; }
    const GpuArrayView& view() const { return _view; }

  private:
    int _count = 0;
    GpuArrayView _view;
};

void launchRigidStateToMat4CUDA(const GpuArrayView& rigidState,
                                GpuArrayView& transforms, int count);

void launchIndexedRigidStateToMat4CUDA(const GpuArrayView& rigidState,
                                       const GpuArrayView& rigidRows,
                                       GpuArrayView& transforms);

// Compose selected actor poses with column-major local matrices directly in
// each part's mapped renderer buffer, without intermediate GPU matrices.
void launchRigidStateToMappedMat4CUDA(
    const GpuArrayView& rigidState, const GpuArrayView& rigidRows,
    const std::vector<std::array<float, 16>>& localTransforms,
    const std::vector<GpuArrayView>& mappedTransforms);

// Gather selected articulation rows into link-major renderer transforms:
// [articulation, link, 13] -> [link, selected_articulation, 4, 4].
void launchArticulationLinkStateToMat4CUDA(
    const GpuArrayView& articulationLinkState,
    const GpuArrayView& articulationRows, const GpuArrayView& linkIndices,
    GpuArrayView& transforms, int linkCount);

void launchArticulationLinkStateToMappedMat4CUDA(
    const GpuArrayView& articulationLinkState,
    const GpuArrayView& articulationRows, const GpuArrayView& linkIndices,
    const std::vector<GpuArrayView>& mappedTransforms);

} // namespace Sim
} // namespace KE

#endif

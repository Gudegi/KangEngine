#pragma once

// CPU capture data and CUDA image handles for external renderers such as ovrtx.

#include <cstdint>
#include <vector>

namespace KE {

// GPU output from backends such as ovrtx, borrowed for window presentation.
// Display-encoded RGBA8 with top-down rows. Valid until
// the next frame, resize, readback, or backend release; wait before reading.
struct CudaRenderImage {
    int width = 0, height = 0;
    int device = 0;
    uintptr_t array = 0;
    uintptr_t readyEvent = 0;
};

struct RenderImage {
    int width = 0;
    int height = 0;
    // Packed RGB8 after configured tone mapping and gamma; no row padding.
    std::vector<uint8_t> pixels;
};

} // namespace KE

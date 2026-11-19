#pragma once

// Owned CPU image for capture and saving, independent of GPU resources.

#include <cstdint>
#include <vector>

namespace KE {

struct RenderImage {
    int width = 0;
    int height = 0;
    // Packed RGB8 after configured tone mapping and gamma; no row padding.
    std::vector<uint8_t> pixels;
};

} // namespace KE

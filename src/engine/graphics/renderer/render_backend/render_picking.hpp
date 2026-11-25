#pragma once

// Matches screen-picking responses to their submitted view and frame.

#include "engine/graphics/renderer/renderer_types.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

namespace KE {

struct RenderPickRequest {
    uint64_t id = 0;
    uint64_t frame = 0;
    // OpenGL NDC: bottom-left (-1, -1), depth [-1, 1].
    glm::vec2 ndc{0.0f};
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    int width = 0;
    int height = 0;
};

struct RenderPickResult {
    uint64_t requestId = 0;
    uint64_t frame = 0;
    RenderPickHit hit;
};

} // namespace KE

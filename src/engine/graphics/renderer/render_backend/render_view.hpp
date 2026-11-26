#pragma once

// Viewport inspection outputs shared by render backends.

namespace KE {

enum class RenderViewMode { RGB, Depth, Normals, ObjectIds };

struct RenderViewSettings {
    RenderViewMode mode = RenderViewMode::RGB;
    // Positive camera-forward depth in scene units; black background, white far limit.
    float depthMax = 10.0f;
};

} // namespace KE

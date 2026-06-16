#ifndef _RENDERER_TYPES_HPP_
#define _RENDERER_TYPES_HPP_

#include <cstdint>

#include <glm/vec3.hpp>

#include "geometry/bounds.hpp"

namespace KE {

namespace Scene {
class Prim;
} // namespace Scene

using RenderableHandle = uint32_t;
using MeshHandle = RenderableHandle; // Deprecated: use RenderableHandle.
static constexpr RenderableHandle InvalidHandle = ~0u;

enum class TransformSource {
    SceneGraph,     // Prim/scene graph owns transforms.
    ExternalBuffer, // Caller owns per-instance transform arrays.
};

struct RayPickResult {
    bool hit = false;
    RenderableHandle handle = InvalidHandle;
    int instanceIndex = -1;
    TransformSource transformSource = TransformSource::SceneGraph;
    Scene::Prim* prim = nullptr;
    float distance = 0.0f;
    glm::vec3 position = glm::vec3(0.0f);
    Geometry::AABB bounds;
};

} // namespace KE

#endif

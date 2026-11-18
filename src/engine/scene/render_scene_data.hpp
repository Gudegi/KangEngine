#pragma once

// Shared mesh sources and instance state used by scene-to-render adapters.

#include "geometry/mesh_data.hpp"
#include "engine/scene/component/resource_component.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace KE {
namespace Scene {

using RenderInstanceId = uint64_t;
constexpr RenderInstanceId InvalidRenderInstance = 0;

// CPU scene values shared with render adapters; no Prim or GPU resource
// pointers.
struct RenderInstanceData {
    std::vector<glm::mat4> transforms;
    std::vector<glm::vec4> colors;
    std::vector<RenderInstanceId> ids;
    // Refreshed on query; rendering does not need selection policy.
    std::vector<uint8_t> pickable;
};

// Shared source mesh and its scene-local resource identity.
struct RenderMeshData {
    std::shared_ptr<const MeshData> mesh;
    // SceneResourceManager identity; invalid for legacy unregistered sources.
    ResourceHandle resource = InvalidResourceHandle;
};

} // namespace Scene
} // namespace KE

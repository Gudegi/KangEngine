#pragma once

// Shared batch and scene inputs consumed by render backends.

#include "geometry/mesh_data.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/render/render_light_data.hpp"
#include "engine/scene/render/render_material_data.hpp"

#include <cstdint>
#include <memory>
#include <string>
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

// Owned mesh-local deformation; topology and source MeshData stay unchanged.
struct RenderGeometryData {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
};

// One object's source resources and evaluated world state.
struct RenderObjectSource {
    RenderInstanceId id = InvalidRenderInstance;
    std::string path;
    RenderMeshData mesh;
    std::shared_ptr<const RenderGeometryData> geometry;
    RenderMaterialData material;
    ResourceHandle materialResource = InvalidResourceHandle;
    RenderTextureSources textures;
    glm::mat4 world{1.0f};
    glm::vec4 color{1.0f};
    bool visible = true;
    bool pickable = true;
    bool doubleSided = false;
    bool castsShadow = true;
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
};

struct RenderCameraSource {
    glm::mat4 world{1.0f};
    float verticalFov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
};

// Full scene state or an incremental update consumed by either render adapter.
struct RenderSceneFrame {
    std::vector<RenderObjectSource> objects;
    RenderCameraSource camera;
    RenderLightData lighting;
    // Incremental frames upsert objects and remove only the listed identities.
    bool incremental = false;
    std::vector<RenderInstanceId> removedObjects;
    bool lightingChanged = true;
};

} // namespace Scene
} // namespace KE

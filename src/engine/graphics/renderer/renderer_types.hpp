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

namespace RendererTextureSlot {

constexpr int BaseColor = 0;
constexpr int Diffuse =
    BaseColor; // Compatibility name for Phong/common shaders.
constexpr int Shadow0 = 1;
constexpr int Shadow1 = 2;
constexpr int Shadow2 = 3;
constexpr int Shadow3 = 4;
constexpr int Normal = 5;
constexpr int MetallicRoughness = 6;
constexpr int AmbientOcclusion = 7;
constexpr int Emissive = 8;
constexpr int Metallic = 9;
constexpr int Roughness = 10;
constexpr int OcclusionRoughnessMetallic = 11;

} // namespace RendererTextureSlot

enum class TextureRole {
    BaseColor,
    Diffuse = BaseColor,
    Normal,
    MetallicRoughness,
    AmbientOcclusion,
    Emissive,
    Metallic,
    Roughness,
    OcclusionRoughnessMetallic,
};

constexpr int textureRoleSlot(TextureRole role) {
    switch (role) {
    case TextureRole::BaseColor:
        return RendererTextureSlot::BaseColor;
    case TextureRole::Normal:
        return RendererTextureSlot::Normal;
    case TextureRole::MetallicRoughness:
        return RendererTextureSlot::MetallicRoughness;
    case TextureRole::AmbientOcclusion:
        return RendererTextureSlot::AmbientOcclusion;
    case TextureRole::Emissive:
        return RendererTextureSlot::Emissive;
    case TextureRole::Metallic:
        return RendererTextureSlot::Metallic;
    case TextureRole::Roughness:
        return RendererTextureSlot::Roughness;
    case TextureRole::OcclusionRoughnessMetallic:
        return RendererTextureSlot::OcclusionRoughnessMetallic;
    }
    return RendererTextureSlot::BaseColor;
}

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

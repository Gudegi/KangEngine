#ifndef _RENDERER_TYPES_HPP_
#define _RENDERER_TYPES_HPP_

#include <cstdint>
#include <functional>
#include <utility>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "engine/graphics/material/colors.hpp"
#include "geometry/bounds.hpp"
#include "sim/gpu_array_view.hpp"

namespace KE {

namespace Backend {
class BindGroup;
class RenderPassEncoder;
class RenderTarget;
} // namespace Backend

namespace Scene {
class Prim;
} // namespace Scene

using RenderableHandle = uint32_t;
static constexpr RenderableHandle InvalidHandle = ~0u;

// Lightweight RHI escape hatch for examples and renderer experiments. Hooks
// only record draw commands; create pipelines and resources outside callbacks.
enum class RenderHookPhase : uint8_t {
    AfterOpaque,
    AfterTransparent,
};

using RenderHookHandle = uint64_t;
static constexpr RenderHookHandle InvalidRenderHook = 0;

struct RenderHookContext {
    Backend::RenderPassEncoder& pass;
    Backend::RenderTarget& target;
    Backend::BindGroup* frameBindings = nullptr;
    int width = 0;
    int height = 0;
};

using RenderHookCallback = std::function<void(RenderHookContext&)>;

enum class TransformSource {
    SceneGraph,     // Prim/scene graph owns transforms.
    ExternalBuffer, // Caller owns per-instance transform arrays.
};

// Controls how fragment alpha affects rasterization. Mask keeps normal opaque
// depth behavior and rejects fragments below a cutoff; Blend uses the
// transparent pass. Opaque remains the default for backwards compatibility.
enum class AlphaMode {
    Opaque,
    Mask,
    Blend,
};

struct BackgroundSettings {
    glm::vec4 checkerColor1 = ColorLibrary::getVec4(ColorType::WHITE);
    glm::vec4 checkerColor2 = ColorLibrary::getVec4(ColorType::DARK_GREEN);
    bool showGrid = true;
    glm::vec4 gridColor = ColorLibrary::getVec4(ColorType::BLACK);
    glm::vec4 backgroundColor = {0.2f, 0.3f, 0.3f, 1.0f};
};

enum class ExternalBufferFormat {
    Mat4, // float32 column-major matrices, one per instance
    PositionRotation,
    PositionRotationScale,
    Custom,
};

enum class ExternalSyncPolicy {
    None,
    Versioned,
    Fence,
    Event,
};

// External transform source attached to a renderable. The descriptor is a view:
// memory ownership and synchronization are provided by GpuArrayView metadata.
struct ExternalBufferDesc {
    Sim::GpuArrayView view;
    ExternalBufferFormat format = ExternalBufferFormat::Mat4;
    int count = 0;       // 0 derives the count from view.shape[0].
    int strideBytes = 0; // 0 derives the stride from view metadata.
    ExternalSyncPolicy syncPolicy = ExternalSyncPolicy::None;
};

// Wraps a CPU/CUDA Mat4 view as a renderer-owned external transform source.
inline ExternalBufferDesc makeExternalMat4BufferDesc(
    Sim::GpuArrayView view, int count = 0,
    ExternalSyncPolicy syncPolicy = ExternalSyncPolicy::Versioned) {
    view.dtype = Sim::SimDType::Float32;
    if (count > 0 && view.shape.empty())
        view.shape = {count, 4, 4};
    if (view.strides.empty())
        view.strides = {16, 4, 1};

    ExternalBufferDesc desc;
    desc.view = std::move(view);
    desc.format = ExternalBufferFormat::Mat4;
    desc.count = count;
    desc.strideBytes = 0;
    desc.syncPolicy = syncPolicy;
    return desc;
}

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
constexpr int Specular = 12;
constexpr int Alpha = 13;

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

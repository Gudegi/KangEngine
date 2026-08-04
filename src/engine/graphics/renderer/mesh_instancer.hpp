#ifndef _MESH_INSTANCER_HPP_
#define _MESH_INSTANCER_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/native/prim.hpp"
#include "geometry/bounds.hpp"
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace KE {

namespace RendererAttribute {

// Vertex attribute locations for MeshInstancer shaders.
// Keep these in sync with GLSL layout(location = ...).
constexpr int Position = 0;
constexpr int Normal = 1;
constexpr int TexCoord = 2;
constexpr int InstanceTransform0 = 3;
constexpr int InstanceTransform1 = 4;
constexpr int InstanceTransform2 = 5;
constexpr int InstanceTransform3 = 6;
constexpr int InstanceColor = 7;
constexpr int BoneIndices = 8;
constexpr int BoneWeights = 9;
constexpr int Tangent = 10;

} // namespace RendererAttribute

// Pipeline vertex-buffer array indices. These are intentionally distinct from
// RendererAttribute shader locations above: one buffer slot may provide
// several shader attributes (InstanceTransform provides locations 3-6).
enum class MeshVertexBufferSlot : uint32_t {
    Position = 0,
    InstanceTransform,
    Normal,
    TexCoord,
    InstanceColor,
    Tangent,
    BoneIndices,
    BoneWeights,
};

constexpr uint32_t vertexBufferSlot(MeshVertexBufferSlot slot) {
    return static_cast<uint32_t>(slot);
}

// Instanced renderer for one mesh/material combination.
class MeshInstancer {

  private:
    struct VertexBufferBinding {
        Backend::Buffer* buffer = nullptr;
        Backend::VertexAttribute attribute{};
    };

    Backend::GraphicsDevice* _device = nullptr;
    Material* _material = nullptr;

    // Geometry (static) — shared by all instances
    std::vector<std::unique_ptr<Backend::Buffer>> _vbos;
    std::unique_ptr<Backend::Buffer> _indexBuffer;
    std::unique_ptr<Backend::Buffer> _overrideTransformVBO;
    std::unique_ptr<Backend::Buffer> _boneMatricesUBO;
    std::unique_ptr<Backend::Buffer> _alphaParamsUBO;
    int _numIndices = 0;

    // Instance data (dynamic, uploaded each frame)
    std::unique_ptr<Backend::Buffer> _transformVBO; // mat4 × N  (loc 3-6)
    std::unique_ptr<Backend::Buffer> _colorVBO;     // vec4 × N  (loc 7)
    int _allocatedInstances = 0;
    int _visibleCount = 0;
    bool _hasTransparent = false;
    bool _colorsDirty = false;
    bool _useExternalTransforms = false;
    bool _doubleSided = false;
    bool _castsShadow = true;
    AlphaMode _alphaMode = AlphaMode::Opaque;
    float _alphaCutoff = 0.5f;
    bool _hasSkinning = false;
    bool _hasTangents = false;
    TransformSource _transformSource{};
    Geometry::AABB _localBounds;
    Geometry::Sphere _localSphere;
    Geometry::AABB _combinedWorldBounds;
    std::vector<Geometry::AABB> _worldBounds;
    std::vector<glm::mat4> _transforms;
    std::vector<glm::vec4> _colors;
    std::vector<glm::mat4> _culledTransforms;
    std::vector<glm::vec4> _culledColors;
    std::vector<std::pair<Backend::Texture*, int>> _textures;
    std::vector<glm::mat4> _boneMatrices;
    std::vector<Scene::Prim*> _instancePrims;
    std::vector<VertexBufferBinding> _vertexBufferBindings;
    ExternalBufferDesc _externalBufferDesc;
    bool _hasExternalBufferDesc = false;
    bool _externalBufferLoaded = false;
    bool _usesGpuExternalTransforms = false;
    bool _hasDirectCudaTransforms = false;
    uint64_t _externalBufferVersion = 0;

    std::vector<Scene::Prim*> _prims;

    void _initMeshData(const Scene::MeshData& mesh);
    void _setupSkinningAttribs(const Scene::SkinnedMeshData& skinnedMesh);
    void _initOverrideInstanceData();
    // Recreate instance VBOs when transform/color capacity grows.
    void _reallocate(int newMax);
    // Cache per-instance world AABBs used by frustum culling/debug stats.
    void _updateWorldBounds(const std::vector<glm::mat4>& transforms);
    bool _hasVisibleOwnerPrim() const;
    // Upload the currently drawable instance transform/color buffer to GPU.
    void _uploadInstanceData(const std::vector<glm::mat4>& transforms,
                             const std::vector<glm::vec4>& colors);
    // Copy a CPU external view into the backend instance buffer. GPU views are
    // rejected until the active graphics backend provides explicit interop.
    void _consumeExternalBuffer();
    void _uploadOverrideTransform(const glm::mat4& transform);
    void _updateTransparency();
    void _updateAlphaParamsBuffer();
    void _recordGeometry(Backend::RenderPassEncoder& pass,
                         Backend::Buffer* transformBuffer,
                         uint32_t instanceCount, bool includeTexCoord,
                         bool includeSkinning) const;

  public:
    MeshInstancer() = default;
    MeshInstancer(MeshInstancer&&) = default;
    MeshInstancer& operator=(MeshInstancer&&) = default;
    MeshInstancer(const MeshInstancer&) = delete;
    MeshInstancer& operator=(const MeshInstancer&) = delete;

    // Upload static geometry. Must be called before addPrim/update/render.
    void init(Backend::GraphicsDevice* device, const Scene::MeshData& mesh,
              TransformSource transformSource,
              Material* material = nullptr);
    void init(Backend::GraphicsDevice* device,
              const Scene::SkinnedMeshData& skinnedMesh,
              TransformSource transformSource, Material* material = nullptr);

    void addPrim(Scene::Prim* prim);
    void removePrim(Scene::Prim* prim);

    // Collect visible prim transforms + colors, upload to instance VBOs.
    // Call once per frame before render().
    void update();

    // Track B mode: directly upload transforms (and optionally colors), then
    // keep using that external instance buffer instead of polling Prims.
    // Owner Prim visibility is a batch-level toggle; per-instance visibility is
    // intentionally not supported here.
    // colors == nullptr: skip color upload (use previously set colors).
    void updateFromTransforms(const std::vector<glm::mat4>& transforms,
                              const std::vector<glm::vec4>* colors = nullptr);
    void setExternalBuffer(const ExternalBufferDesc& desc);
    void prepareDirectCudaTransforms(int count);
    Backend::Buffer* transformBuffer() { return _transformVBO.get(); }
    bool hasExternalBuffer() const { return _hasExternalBufferDesc; }
    const ExternalBufferDesc& externalBuffer() const {
        return _externalBufferDesc;
    }

    // One-time color upload. Call once in setup instead of passing colors every
    // frame.
    void setColors(const std::vector<glm::vec4>& colors);

    // Update vertex positions and normals for deformable meshes (cloth, soft
    // body). Vertex count must match the mesh passed to init().
    void updateGeometry(const std::vector<glm::vec3>& positions,
                        const std::vector<glm::vec3>& normals);
    // Store bone matrices for the RHI skinning uniform buffer.
    void updateRenderableSkinningMatrices(
        const std::vector<glm::mat4>& boneMatrices);
    // Compact instance buffers to objects intersecting the current frustum.
    void applyFrustumCulling(const Geometry::Frustum* frustum);
    bool findRayIntersection(const Geometry::Ray& ray, int& outInstanceIndex,
                             float& outDistance,
                             Geometry::AABB* outBounds = nullptr,
                             Scene::Prim** outPrim = nullptr) const;
    bool findPrimInstance(Scene::Prim* prim, int& outInstanceIndex,
                          Geometry::AABB* outBounds = nullptr) const;
    bool getInstanceTransform(int instanceIndex, glm::mat4& outTransform) const;
    bool setInstanceTransform(int instanceIndex, const glm::mat4& transform);
    TransformSource transformSource() const { return _transformSource; }

    // Record the currently visible instance batch without touching backend
    // state. Resources are only consumed when the command buffer is submitted.
    void recordDraw(Backend::RenderPassEncoder& pass,
                    bool includeTexCoord = false,
                    bool includeSkinning = false) const;
    void recordForwardDraw(Backend::RenderPassEncoder& pass) const;
    void recordSkinnedForwardDraw(Backend::RenderPassEncoder& pass) const;
    // Full static material geometry used by Phong/PBR pipelines. In addition
    // to the common forward inputs this records UV and tangent streams.
    void recordMaterialDraw(Backend::RenderPassEncoder& pass,
                            bool requireTexCoord,
                            bool requireTangent,
                            bool includeSkinning = false) const;
    void recordSkinnedMaterialDraw(Backend::RenderPassEncoder& pass,
                                   bool requireTangent) const;
    // Record the same one-instance geometry draw through the RHI. The caller's
    // pipeline must use MeshVertexBufferSlot::Position and
    // MeshVertexBufferSlot::InstanceTransform. The transform buffer provides
    // shader attribute locations 3-6.
    void recordInstanceMask(Backend::RenderPassEncoder& pass,
                            int instanceIndex, bool includeTexCoord = false,
                            bool includeSkinning = false);
    Backend::Buffer* boneMatricesBuffer() const {
        return _boneMatricesUBO.get();
    }
    Backend::Texture* alphaMaskTexture() const;
    Backend::Buffer* alphaParamsBuffer() const {
        return _alphaParamsUBO.get();
    }
    bool alphaMaskUsesRedChannel() const {
        return _material && _material->alphaTextureUsesRedChannel();
    }

    // DoubleSided means the mesh can be seen both back and forward side.
    void setDoubleSided(bool v) { _doubleSided = v; }
    bool isDoubleSided() const { return _doubleSided; }
    void setCastsShadow(bool v) { _castsShadow = v; }
    bool castsShadow() const { return _castsShadow; }
    void setAlphaMode(AlphaMode mode, float cutoff = 0.5f);
    AlphaMode alphaMode() const { return _alphaMode; }
    float alphaCutoff() const { return _alphaCutoff; }
    bool hasSkinning() const { return _hasSkinning; }
    bool hasTangents() const { return _hasTangents; }
    const Geometry::AABB& localBounds() const { return _localBounds; }
    const Geometry::Sphere& localSphere() const { return _localSphere; }
    const Geometry::AABB& combinedWorldBounds() const {
        return _combinedWorldBounds;
    }
    const std::vector<Geometry::AABB>& worldBounds() const {
        return _worldBounds;
    }

    void setTexture(Backend::Texture* tex, int slot = 0) {
        for (auto& [t, s] : _textures) {
            if (s == slot) {
                t = tex;
                return;
            }
        }
        _textures.emplace_back(tex, slot);
    }
    Backend::Texture* textureAtSlot(int slot) const {
        for (const auto& [texture, textureSlot] : _textures)
            if (textureSlot == slot)
                return texture;
        return nullptr;
    }
    Material* material() const { return _material; }
    bool hasTransparent() const { return _hasTransparent; }
    int instanceCount() const { return static_cast<int>(_transforms.size()); }
    int visibleCount() const { return _visibleCount; }
};

} // namespace KE

#endif

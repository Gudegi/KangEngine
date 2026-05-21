#ifndef _MESH_INSTANCER_HPP_
#define _MESH_INSTANCER_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/native/prim.hpp"
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
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

} // namespace RendererAttribute

// Instanced renderer for a single (shader, mesh) combination.
// All Prims sharing the same MeshData pointer + Shader are batched
// into one glDrawElementsInstanced call.
class MeshInstancer {

  private:
    Backend::GraphicsDevice* _device = nullptr;
    Backend::Shader* _shader = nullptr;
    PhongMaterial* _material = nullptr;

    // Geometry (static) — shared by all instances
    std::unique_ptr<Backend::VertexArray> _vao;
    std::vector<std::unique_ptr<Backend::Buffer>> _vbos;
    std::unique_ptr<Backend::Buffer> _indexBuffer;
    int _numIndices = 0;

    // Instance data (dynamic, uploaded each frame)
    std::unique_ptr<Backend::Buffer> _transformVBO; // mat4 × N  (loc 3-6)
    std::unique_ptr<Backend::Buffer> _colorVBO;     // vec4 × N  (loc 7)
    int _allocatedInstances = 0;
    int _visibleCount = 0;
    bool _hasTransparent = false;
    bool _useExternalTransforms = false;
    bool _doubleSided = false;
    bool _castsShadow = true;
    bool _hasSkinning = false;
    std::vector<std::pair<Backend::Texture*, int>> _textures;
    std::vector<glm::mat4> _boneMatrices;

    std::vector<Scene::Prim*> _prims;

    void _initMeshData(const Scene::MeshData& mesh);
    void _setupSkinningAttribs(const Scene::SkinnedMeshData& skinnedMesh);
    void _setupInstanceAttribs();
    void _reallocate(int newMax);

  public:
    MeshInstancer() = default;
    MeshInstancer(MeshInstancer&&) = default;
    MeshInstancer& operator=(MeshInstancer&&) = default;
    MeshInstancer(const MeshInstancer&) = delete;
    MeshInstancer& operator=(const MeshInstancer&) = delete;

    // Upload static geometry. Must be called before addPrim/update/render.
    void init(Backend::GraphicsDevice* device, Backend::Shader* shader,
              const Scene::MeshData& mesh, PhongMaterial* material = nullptr);
    void init(Backend::GraphicsDevice* device, Backend::Shader* shader,
              const Scene::SkinnedMeshData& skinnedMesh,
              PhongMaterial* material = nullptr);

    void addPrim(Scene::Prim* prim);
    void removePrim(Scene::Prim* prim);

    // Collect visible prim transforms + colors, upload to instance VBOs.
    // Call once per frame before render().
    void update();

    // Track B mode: directly upload transforms (and optionally colors), then
    // keep using that external instance buffer instead of polling Prims.
    // colors == nullptr: skip color upload (use previously set colors).
    void updateFromTransforms(const std::vector<glm::mat4>& transforms,
                              const std::vector<glm::vec4>* colors = nullptr);

    // One-time color upload. Call once in setup instead of passing colors every
    // frame.
    void setColors(const std::vector<glm::vec4>& colors);

    // Update vertex positions and normals for deformable meshes (cloth, soft
    // body). Vertex count must match the mesh passed to init().
    void updateGeometry(const std::vector<glm::vec3>& positions,
                        const std::vector<glm::vec3>& normals);
    void updateSkinningMatrices(const std::vector<glm::mat4>& boneMatrices);
    void uploadSkinningMatrices(Backend::Shader* shader = nullptr);

    void render();

    // DoubleSided means the mesh can be seen both back and forward side.
    void setDoubleSided(bool v) { _doubleSided = v; }
    bool isDoubleSided() const { return _doubleSided; }
    void setCastsShadow(bool v) { _castsShadow = v; }
    bool castsShadow() const { return _castsShadow; }
    bool hasSkinning() const { return _hasSkinning; }

    void setTexture(Backend::Texture* tex, int slot = 0) {
        for (auto& [t, s] : _textures) {
            if (s == slot) { t = tex; return; }
        }
        _textures.emplace_back(tex, slot);
    }
    void bindTextures() const {
        for (auto& [tex, slot] : _textures)
            if (tex) tex->bind(slot);
    }

    Backend::Shader* shader() const { return _shader; }
    PhongMaterial* material() const { return _material; }
    bool hasTransparent() const { return _hasTransparent; }
    int visibleCount() const { return _visibleCount; }
};

} // namespace KE

#endif

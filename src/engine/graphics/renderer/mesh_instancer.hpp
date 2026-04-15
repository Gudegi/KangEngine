#ifndef _MESH_INSTANCER_HPP_
#define _MESH_INSTANCER_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "scene/scene_backend.hpp"
#include "scene/native/prim.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace KE {

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

    std::vector<Scene::Prim*> _prims;

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

    void addPrim(Scene::Prim* prim);

    // Collect visible prim transforms + colors, upload to instance VBOs.
    // Call once per frame before render().
    void update();

    void render();

    Backend::Shader* shader() const { return _shader; }
    PhongMaterial* material() const { return _material; }
    bool hasTransparent() const { return _hasTransparent; }
    int visibleCount() const { return _visibleCount; }
};

} // namespace KE

#endif

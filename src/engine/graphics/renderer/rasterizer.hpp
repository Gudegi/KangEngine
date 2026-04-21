#ifndef _RASTERIZER_HPP_
#define _RASTERIZER_HPP_

#include "engine/graphics/renderer/renderer.hpp"
#include "engine/graphics/renderer/mesh_instancer.hpp"
#include "engine/graphics/renderer/light.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/graphics/material/material.hpp"
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include "utils/types.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace KE {

using MeshHandle = uint32_t;
static constexpr MeshHandle InvalidHandle = ~0u;

// ---------------------------------------------------------------------------
// Two-track rendering
//
// Track A — Prim-based (scene graph drives transforms)
//   addShape(shader, prim) → instancer polls Prim attributes every frame
//   Use for: static geometry, any object the scene graph owns
//
// Track B — Handle-based (caller drives transforms)
//   handle = addShape(shader, prim)       // register once at setup
//   setShapeColors(handle, colors)        // upload colors once
//   updateShapeTransforms(handle, mats)   // upload transforms per frame
//   Use for: PhysX rigid bodies, large instanced crowds, anything with
//            per-frame external transform arrays
//
// The two tracks share the same instancer; switching is per-instancer via
// the _externalUpdate flag in MeshInstancer.
// ---------------------------------------------------------------------------

class Rasterizer : public Renderer {
  private:
    // Prim-based instanced rendering
    struct InstancerKey {
        Backend::Shader* shader;
        const Scene::MeshData* mesh;
        PhongMaterial* material; // nullptr for shader-only path
        bool operator<(const InstancerKey& o) const {
            if (shader != o.shader)
                return shader < o.shader;
            if (mesh != o.mesh)
                return mesh < o.mesh;
            return material < o.material;
        }
    };
    std::map<InstancerKey, MeshInstancer> _instancers;

    std::map<InstancerKey, MeshHandle> _handleMap;
    std::vector<MeshInstancer*> _handleTable;

    std::unique_ptr<Backend::Buffer> _cameraUBO;
    std::unique_ptr<Backend::Buffer> _lightUBO;
    DirectionalLight _light;
    bool _lightDirty = true;

  public:
    Rasterizer(Backend::GraphicsDevice* graphicsDevice);

    void setLight(const DirectionalLight& light) {
        _light = light;
        _lightDirty = true;
    }
    const DirectionalLight& getLight() const { return _light; }

    MeshHandle addShape(Backend::Shader* shader, Scene::Prim* prim);
    MeshHandle addShape(PhongMaterial* material, Scene::Prim* prim);
    void removePrim(MeshHandle handle, Scene::Prim* prim);

    void updateShapeTransforms(MeshHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors = nullptr);

    void setShapeColors(MeshHandle handle,
                        const std::vector<glm::vec4>& colors);

    void setSkybox(const std::string& path, UpAxis upAxis = UpAxis::Y) {
        _graphicsDevice->setSkybox(path, upAxis);
    }
    void setSkybox(const std::vector<std::string>& paths,
                   UpAxis upAxis = UpAxis::Y) {
        _graphicsDevice->setSkybox(paths, upAxis);
    }

    void render(const glm::mat4& view, const glm::mat4& proj) override;
};

} // namespace KE

#endif

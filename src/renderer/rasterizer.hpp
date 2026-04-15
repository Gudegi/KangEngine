#ifndef _RASTERIZER_HPP_
#define _RASTERIZER_HPP_

#include "renderer/renderer.hpp"
#include "renderer/mesh_instancer.hpp"
#include "backend/base/graphics_device.hpp"
#include "scene/scene_backend.hpp"
#include "material/material.hpp"
#include <glm/mat4x4.hpp>
#include "utils/types.hpp"
#include <map>
#include <memory>
#include <string>

namespace KE {

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

    std::unique_ptr<Backend::Buffer> _cameraUBO;
    std::unique_ptr<Backend::Buffer> _lightUBO;

  public:
    Rasterizer(Backend::GraphicsDevice* graphicsDevice,
               Scene::SceneBackend* scene);

    size_t addShape(Backend::Shader* shader, Scene::Prim* prim);
    size_t addShape(PhongMaterial* material, Scene::Prim* prim);

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

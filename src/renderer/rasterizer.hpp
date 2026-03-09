#ifndef _RASTERIZER_HPP_
#define _RASTERIZER_HPP_

#include "renderer/renderer.hpp"
#include "backend/base/graphics_device.hpp"
#include "scene/scene_backend.hpp"
#include "material/material.hpp"
#include <glm/mat4x4.hpp>
#include <memory>
#include <vector>

namespace KE {
class Rasterizer : public Renderer {
    std::vector<std::shared_ptr<Scene::ShapeRenderBuffer>> _renderList;

  public:
    Rasterizer(Backend::GraphicsDevice* graphicsDevice,
               Scene::SceneBackend* scene);

    std::shared_ptr<Scene::ShapeRenderBuffer>
    createRenderBuffer(Backend::Shader* shader,
                       const std::shared_ptr<Scene::MeshData>& meshData);

    size_t addShape(Backend::Shader* shader,
                    std::shared_ptr<Scene::MeshData> meshData);
    size_t addShape(PhongMaterial* material,
                    std::shared_ptr<Scene::MeshData> meshData);
    size_t addShape(Backend::Shader* shader, Scene::Prim* prim);

    void render(const glm::mat4& view, const glm::mat4& proj) override;
};
} // namespace KE

#endif
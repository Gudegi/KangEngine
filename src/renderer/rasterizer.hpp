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

    std::unique_ptr<Backend::Buffer> _cameraUBO;
    std::unique_ptr<Backend::Buffer> _lightUBO;

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

    /*
    // Render outlined shapes with given shader
    void renderOutline(const glm::mat4& view, const glm::mat4& proj,
                       Backend::Shader* outlineShader, float thickness,
                       const glm::vec4& color);

    // Mark a shape as excluded from outline rendering (e.g. ground plane)
    void setOutlined(size_t idx, bool outlined);
    */
};
} // namespace KE

#endif
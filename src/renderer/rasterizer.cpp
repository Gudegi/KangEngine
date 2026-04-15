#include "rasterizer.hpp"
#include "backend/base/graphics_device.hpp"
#include "material/material.hpp"
#include "scene/native/prim.hpp"
#include "scene/scene_backend.hpp"
#include <glm/glm.hpp>
#include <memory>

constexpr int CAMERA_UBO_BIND_SLOT = 0;
constexpr int LIGHT_UBO_BIND_SLOT = 1;

namespace KE {

Rasterizer::Rasterizer(Backend::GraphicsDevice* graphicsDevice,
                       Scene::SceneBackend* scene) {
    _graphicsDevice = graphicsDevice;
    _scene = scene;
    _cameraUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                               2 * sizeof(glm::mat4));
    _graphicsDevice->bindUniformBuffer(_cameraUBO.get(), CAMERA_UBO_BIND_SLOT);
    _lightUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                              sizeof(glm::mat4));
    _graphicsDevice->bindUniformBuffer(_lightUBO.get(), LIGHT_UBO_BIND_SLOT);
}

// Prim-based (instanced)

size_t Rasterizer::addShape(Backend::Shader* shader, Scene::Prim* prim) {
    auto meshData = prim->getMeshData();
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty())
        return static_cast<size_t>(-1);

    InstancerKey key{shader, meshData.get(), nullptr};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, shader, *meshData);
        it = newIt;
    }
    it->second.addPrim(prim);
    return 0;
}

size_t Rasterizer::addShape(PhongMaterial* material, Scene::Prim* prim) {
    auto meshData = prim->getMeshData();
    if (!material || !meshData || meshData->vertices.empty() ||
        meshData->indices.empty())
        return static_cast<size_t>(-1);

    auto* shader = material->getShader();
    InstancerKey key{shader, meshData.get(), material};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, shader, *meshData, material);
        it = newIt;
    }
    it->second.addPrim(prim);
    return 0;
}

// Render

void Rasterizer::render(const glm::mat4& view, const glm::mat4& proj) {
    // Upload camera UBO once per frame
    _cameraUBO->setData(&view, sizeof(view));
    _cameraUBO->setData(&proj, sizeof(proj), sizeof(glm::mat4));

    // Update all instancers
    for (auto& [key, inst] : _instancers)
        inst.update();

    // Pass 1: opaque instanced
    for (auto& [key, inst] : _instancers) {
        if (inst.hasTransparent() || inst.visibleCount() == 0)
            continue;
        if (inst.material())
            inst.material()->bind();
        else
            inst.shader()->use();
        inst.render();
    }

    // Pass 2: transparent instanced TODO: impl OIT
    _graphicsDevice->setBlend(true);
    _graphicsDevice->setBlendFunc(Backend::BlendFactor::SrcAlpha,
                                  Backend::BlendFactor::OneMinusSrcAlpha);
    _graphicsDevice->setDepthWrite(false);
    for (auto& [key, inst] : _instancers) {
        if (!inst.hasTransparent() || inst.visibleCount() == 0)
            continue;
        if (inst.material())
            inst.material()->bind();
        else
            inst.shader()->use();
        inst.render();
    }
    _graphicsDevice->setDepthWrite(true);
    _graphicsDevice->setBlend(false);

    // Skybox: drawn last, fills only pixels with no geometry
    _graphicsDevice->drawSkybox(view, proj);
}

} // namespace KE

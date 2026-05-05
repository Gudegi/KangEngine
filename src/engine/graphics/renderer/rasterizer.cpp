#include "rasterizer.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/asset_path.hpp"
#include "utils/types.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <memory>
#include <vector>

constexpr int CAMERA_UBO_BIND_SLOT = 0;
constexpr int LIGHT_UBO_BIND_SLOT = 1;
constexpr int SHADOW_UBO_BIND_SLOT = 2;

namespace KE {

Rasterizer::Rasterizer(Backend::GraphicsDevice* graphicsDevice) {
    _graphicsDevice = graphicsDevice;
    _cameraUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                               2 * sizeof(glm::mat4));
    _graphicsDevice->bindUniformBuffer(_cameraUBO.get(), CAMERA_UBO_BIND_SLOT);
    _lightUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                              3 * sizeof(glm::vec4));
    _graphicsDevice->bindUniformBuffer(_lightUBO.get(), LIGHT_UBO_BIND_SLOT);
    // mat4 lightSpaceMatrix, vec4 shadowParams (xyz: sun_direction,
    // w:shadow_radius), vec4 shadowInfo (x: shadowExtents, y: shadowMapWH, zw:
    // unused)
    _shadowUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                               sizeof(glm::mat4) +
                                                   2 * sizeof(glm::vec4));
    _graphicsDevice->bindUniformBuffer(_shadowUBO.get(), SHADOW_UBO_BIND_SLOT);

    _shadowFbo = _graphicsDevice->createFramebuffer(
        {_shadowMapWH, _shadowMapWH, true, false, 0}); // depth-only, no MSAA
    _shadowShader = _graphicsDevice->createShaderFromFile(
        KE::getAssetPath("shaders/shadow.vs"),
        KE::getAssetPath("shaders/shadow.fs"));
    updateShadowUBO(0.0f);
}

// Prim-based (instanced)

MeshHandle Rasterizer::addShape(Backend::Shader* shader, Scene::Prim* prim) {
    auto meshData = prim->getMeshData();
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty())
        return InvalidHandle;

    InstancerKey key{shader, meshData.get(), nullptr};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, shader, *meshData);
        it = newIt;
    }
    it->second.addPrim(prim);

    auto hIt = _handleMap.find(key);
    if (hIt == _handleMap.end()) {
        MeshHandle h = static_cast<MeshHandle>(_handleTable.size());
        _handleMap[key] = h;
        _handleTable.push_back(&it->second);
        return h;
    }
    return hIt->second;
}

MeshHandle Rasterizer::addShape(PhongMaterial* material, Scene::Prim* prim) {
    auto meshData = prim->getMeshData();
    if (!material || !meshData || meshData->vertices.empty() ||
        meshData->indices.empty())
        return InvalidHandle;

    auto* shader = material->getShader();
    InstancerKey key{shader, meshData.get(), material};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, shader, *meshData, material);
        it = newIt;
    }
    it->second.addPrim(prim);

    auto hIt = _handleMap.find(key);
    if (hIt == _handleMap.end()) {
        MeshHandle h = static_cast<MeshHandle>(_handleTable.size());
        _handleMap[key] = h;
        _handleTable.push_back(&it->second);
        return h;
    }
    return hIt->second;
}

void Rasterizer::removePrim(MeshHandle handle, Scene::Prim* prim) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->removePrim(prim);
}

void Rasterizer::updateShapeTransforms(MeshHandle handle,
                                       const std::vector<glm::mat4>& transforms,
                                       const std::vector<glm::vec4>* colors) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->updateFromTransforms(transforms, colors);
}

void Rasterizer::setShapeColors(MeshHandle handle,
                                const std::vector<glm::vec4>& colors) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setColors(colors);
}

void Rasterizer::setShapeDoubleSided(MeshHandle handle, bool doubleSided) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setDoubleSided(doubleSided);
}

void Rasterizer::setShapeTexture(MeshHandle handle, Backend::Texture* tex,
                                 int slot) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setTexture(tex, slot);
}

void Rasterizer::updateMeshGeometry(MeshHandle handle,
                                    const std::vector<glm::vec3>& positions,
                                    const std::vector<glm::vec3>& normals) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->updateGeometry(positions, normals);
}

// Render

void Rasterizer::updateFrameData(const glm::mat4& view, const glm::mat4& proj) {
    // Upload camera UBO once per frame
    _cameraUBO->setData(&view, sizeof(view));
    _cameraUBO->setData(&proj, sizeof(proj), sizeof(glm::mat4));

    glm::vec4 viewDir =
        glm::vec4(glm::normalize(glm::mat3(view) * _light.direction), 0.f);
    _lightUBO->setData(&viewDir, sizeof(glm::vec4), 0 * sizeof(glm::vec4));
    if (_lightDirty) {
        // std140 layout: vec4 direction | vec4 color | vec4 ambient
        glm::vec4 lightColor = glm::vec4(_light.color * _light.intensity, 1.f);
        glm::vec4 ambient = glm::vec4(_light.ambient, 0.f);
        _lightUBO->setData(&lightColor, sizeof(glm::vec4),
                           1 * sizeof(glm::vec4));
        _lightUBO->setData(&ambient, sizeof(glm::vec4), 2 * sizeof(glm::vec4));
        _lightDirty = false;
    }

    // Update all instancers (must run before shadow pass AND scene pass)
    for (auto& [key, inst] : _instancers)
        inst.update();
}

void Rasterizer::render(const glm::mat4& view, const glm::mat4& proj) {
    // Bind shadow map and upload shadow uniforms onto each instancer's shader
    const bool hasShadow = (_shadowMap != nullptr && _shadowExtents > 0.0f);
    if (hasShadow)
        _shadowMap->bind(1);

    auto bindShadowSampler = [&](Backend::Shader* sh) {
        if (hasShadow)
            sh->setInt("shadowMap", 1);
    };

    // Pass 1: opaque instanced
    for (auto& [key, inst] : _instancers) {
        if (inst.hasTransparent() || inst.visibleCount() == 0)
            continue;
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(false);
        if (inst.material()) {
            inst.material()->bind();
            bindShadowSampler(inst.shader());
        } else {
            inst.shader()->use();
            bindShadowSampler(inst.shader());
        }
        inst.bindTextures();
        // Re-bind shadow map after bindTextures() to prevent slot 1 conflict
        if (hasShadow)
            _shadowMap->bind(1);
        inst.render();
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(true);
    }
    // Skybox: drawn last, fills only pixels with no geometry
    _graphicsDevice->drawSkybox(view, proj);

    // Pass 2: transparent instanced TODO: impl OIT
    _graphicsDevice->setBlend(true);
    _graphicsDevice->setBlendFunc(Backend::BlendFactor::SrcAlpha,
                                  Backend::BlendFactor::OneMinusSrcAlpha);
    _graphicsDevice->setDepthWrite(false);
    for (auto& [key, inst] : _instancers) {
        if (!inst.hasTransparent() || inst.visibleCount() == 0)
            continue;
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(false);
        if (inst.material())
            inst.material()->bind();
        else
            inst.shader()->use();
        inst.bindTextures();
        inst.render();
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(true);
    }
    _graphicsDevice->setDepthWrite(true);
    _graphicsDevice->setBlend(false);
}

/////////////// Shadow Pass //////////////

void Rasterizer::updateShadowUBO(float activeExtents) {
    glm::vec4 shadowParams{_light.direction, _shadowRadius};
    glm::vec4 shadowInfo{activeExtents, (float)(_shadowMapWH), 0.f, 0.f};
    _shadowUBO->setData(&_lightSpaceMatrix, sizeof(_lightSpaceMatrix));
    _shadowUBO->setData(&shadowParams, sizeof(shadowParams), sizeof(glm::mat4));
    _shadowUBO->setData(&shadowInfo, sizeof(shadowInfo),
                        sizeof(glm::mat4) + sizeof(glm::vec4));
}

void Rasterizer::setShadowMap(Backend::Texture* tex,
                              const glm::mat4& lightSpaceMat, float radius,
                              float extents) {
    _shadowMap = tex;
    _lightSpaceMatrix = lightSpaceMat;
    _shadowRadius = radius;
    _shadowExtents = extents;
    updateShadowUBO(_shadowMap ? _shadowExtents : 0.0f);
}

glm::mat4 Rasterizer::computeLightSpaceMatrix(Camera& camera,
                                              const UpAxis upAxis) {
    const glm::vec3& sunDir = glm::normalize(_light.direction);
    glm::vec3 up =
        (std::abs(sunDir.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    if (upAxis == UpAxis::Z)
        up = (std::abs(sunDir.z) < 0.99f) ? glm::vec3(0, 0, 1)
                                          : glm::vec3(1, 0, 0);
    glm::vec3 sceneCenter = camera.getTargetPos();
    glm::vec3 lightPos = sceneCenter + sunDir * _shadowExtents;
    glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, up);
    glm::mat4 lightProj =
        glm::ortho(-_shadowExtents, _shadowExtents, -_shadowExtents,
                   _shadowExtents, 0.0f, 2.0f * _shadowExtents);
    glm::mat4 lightSpaceMatrix = lightProj * lightView;
    return lightSpaceMatrix;
}

void Rasterizer::renderShadowMap(Camera& camera, UpAxis upAxis,
                                 int viewportWidth, int viewportHeight) {
    if (!_shadowFbo || !_shadowShader || _shadowExtents <= 0.0f) {
        _shadowMap = nullptr;
        updateShadowUBO(0.0f);
        return;
    }

    _lightSpaceMatrix = computeLightSpaceMatrix(camera, upAxis);
    _shadowMap = _shadowFbo->getDepthTexture();
    updateShadowUBO(_shadowMap ? _shadowExtents : 0.0f);

    _shadowFbo->bind();
    _graphicsDevice->setViewport(0, 0, _shadowMapWH, _shadowMapWH);
    _graphicsDevice->clear(0, 0, 0, 1);
    drawShadowCasters();
    _shadowFbo->unbind();
    _graphicsDevice->setViewport(0, 0, viewportWidth, viewportHeight);
}

void Rasterizer::drawShadowCasters() {
    // Set front face culling to prevent peter panning artifact.
    _graphicsDevice->setCullFaceMode(Backend::CullFaceMode::Front);
    _shadowShader->use();
    for (auto& [key, inst] : _instancers) {
        if (inst.visibleCount() == 0)
            continue;
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(false);
        inst.render();
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(true);
    }
    _graphicsDevice->setCullFaceMode(Backend::CullFaceMode::Back);
}

} // namespace KE

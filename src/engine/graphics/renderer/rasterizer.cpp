#include "rasterizer.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/asset_path.hpp"
#include "utils/types.hpp"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <limits>
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
    // w:shadow_radius), vec4 shadowInfo (x: active shadow ortho half-size, y:
    // shadowMapWH, z: PCF samples, w: unused)
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

MeshHandle Rasterizer::addShape(Backend::Shader* shader, Scene::Prim* prim,
                                RenderTrack track) {
    auto meshData = prim->resolveMeshData();
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty())
        return InvalidHandle;

    InstancerKey key{shader, meshData.get(), nullptr, track};
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

MeshHandle Rasterizer::addShape(PhongMaterial* material, Scene::Prim* prim,
                                RenderTrack track) {
    auto meshData = prim->resolveMeshData();
    if (!material || !meshData || meshData->vertices.empty() ||
        meshData->indices.empty())
        return InvalidHandle;

    auto* shader = material->getShader();
    InstancerKey key{shader, meshData.get(), material, track};
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

void Rasterizer::setShapeCastsShadow(MeshHandle handle, bool castsShadow) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setCastsShadow(castsShadow);
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
    const bool hasShadow = (_shadowMap != nullptr && _shadowDistance > 0.0f);
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

void Rasterizer::updateShadowUBO(float activeOrthoHalfSize) {
    glm::vec4 shadowParams{_light.direction, _shadowRadius};
    glm::vec4 shadowInfo{activeOrthoHalfSize, (float)(_shadowMapWH),
                         static_cast<float>(_shadowPcfSamples), 0.f};
    _shadowUBO->setData(&_lightSpaceMatrix, sizeof(_lightSpaceMatrix));
    _shadowUBO->setData(&shadowParams, sizeof(shadowParams), sizeof(glm::mat4));
    _shadowUBO->setData(&shadowInfo, sizeof(shadowInfo),
                        sizeof(glm::mat4) + sizeof(glm::vec4));
}

void Rasterizer::setShadowMap(Backend::Texture* tex,
                              const glm::mat4& lightSpaceMat, float radius,
                              float distance) {
    _shadowMap = tex;
    _lightSpaceMatrix = lightSpaceMat;
    _shadowRadius = radius;
    _shadowDistance = distance;
    _activeShadowOrthoHalfSize = distance;
    updateShadowUBO(_shadowMap ? _activeShadowOrthoHalfSize : 0.0f);
}

glm::mat4 Rasterizer::computeLightSpaceMatrix(Camera& camera,
                                              const UpAxis upAxis) {
    const glm::vec3& sunDir = glm::normalize(_light.direction);
    glm::vec3 up =
        (std::abs(sunDir.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    if (upAxis == UpAxis::Z)
        up = (std::abs(sunDir.z) < 0.99f) ? glm::vec3(0, 0, 1)
                                          : glm::vec3(1, 0, 0);

    /*
    // Fixed-box shadow fitting:
    // glm::vec3 sceneCenter = camera.getTargetPos();
    // glm::vec3 lightPos = sceneCenter + sunDir * _shadowDistance;
    // glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, up);
    // glm::mat4 lightProj =
    //     glm::ortho(-_shadowDistance, _shadowDistance, -_shadowDistance,
    //                _shadowDistance, 0.0f, 2.0f * _shadowDistance);
    // return lightProj * lightView;
    */

    float shadowNear = camera.getNearPlane();
    float shadowFar = std::min(camera.getFarPlane(), _shadowDistance);
    WorldFrustumPos frustumPos = camera.getFrustumPos(shadowNear, shadowFar);
    glm::vec3 corners[] = {
        frustumPos.nearLB, frustumPos.nearLT, frustumPos.nearRB,
        frustumPos.nearRT, frustumPos.farLB,  frustumPos.farLT,
        frustumPos.farRB,  frustumPos.farRT,
    };

    glm::vec3 frustumCenter(0.0f);
    for (const auto& corner : corners)
        frustumCenter += corner;
    frustumCenter /= 8.0f;

    float frustumRadius = 0.0f;
    for (const auto& corner : corners)
        frustumRadius =
            std::max(frustumRadius, glm::length(corner - frustumCenter));
    const float casterMargin = std::max(2.0f, _shadowDistance * 0.25f);
    frustumRadius += casterMargin;
    _activeShadowOrthoHalfSize = frustumRadius;

    glm::vec3 lightPos = frustumCenter + sunDir * frustumRadius;
    glm::mat4 lightView = glm::lookAt(lightPos, frustumCenter, up);

    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const auto& corner : corners) {
        glm::vec3 lightSpaceCorner =
            glm::vec3(lightView * glm::vec4(corner, 1.0f));
        minZ = std::min(minZ, lightSpaceCorner.z);
        maxZ = std::max(maxZ, lightSpaceCorner.z);
    }

    const float zPadding = std::max(5.0f, _shadowDistance * 0.25f);
    // Use a square XY fit to reduce scaling shimmer.
    glm::mat4 lightProj =
        glm::ortho(-frustumRadius, frustumRadius, -frustumRadius, frustumRadius,
                   -maxZ - zPadding, -minZ + zPadding);

    // Snap the projection to the shadow texel grid.
    glm::mat4 lightSpaceMatrix = lightProj * lightView;
    glm::vec4 shadowOrigin =
        lightSpaceMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    float halfSize = static_cast<float>(_shadowMapWH) / 2.0f;
    glm::vec4 shadowOriginPixels = shadowOrigin * halfSize;
    glm::vec4 roundedPixels = glm::round(shadowOriginPixels);
    glm::vec4 offsetPixels = roundedPixels - shadowOriginPixels;

    // Convert texel error back to NDC; depth does not need snapping.
    glm::vec4 offsetNDC = offsetPixels / halfSize;
    offsetNDC.z = 0.0f;
    offsetNDC.w = 0.0f;

    // Shift the projection onto the texel grid.
    lightProj[3] += offsetNDC;

    return lightProj * lightView;
}

void Rasterizer::renderShadowMap(Camera& camera, UpAxis upAxis,
                                 int viewportWidth, int viewportHeight) {
    if (!_shadowFbo || !_shadowShader || _shadowDistance <= 0.0f) {
        _shadowMap = nullptr;
        updateShadowUBO(0.0f);
        return;
    }

    _lightSpaceMatrix = computeLightSpaceMatrix(camera, upAxis);
    _shadowMap = _shadowFbo->getDepthTexture();
    updateShadowUBO(_shadowMap ? _activeShadowOrthoHalfSize : 0.0f);

    _shadowFbo->bind();
    _graphicsDevice->setViewport(0, 0, _shadowMapWH, _shadowMapWH);
    _graphicsDevice->clear(0, 0, 0, 1);
    drawShadowCasters();
    _shadowFbo->unbind();
    _graphicsDevice->setViewport(0, 0, viewportWidth, viewportHeight);
}

void Rasterizer::drawShadowCasters() {
    // Front-face culling avoids storing the same front surfaces that receive
    // the shadow, reducing acne on closed meshes.
    _graphicsDevice->setCullFaceMode(Backend::CullFaceMode::Front);
    _shadowShader->use();
    for (auto& [key, inst] : _instancers) {
        if (inst.visibleCount() == 0)
            continue;
        if (!inst.castsShadow())
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

#include "rasterizer.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/asset_path.hpp"
#include "utils/types.hpp"
#include <algorithm>
#include <array>
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
constexpr int SHADOW_TEXTURE_SLOT_BASE =
    1; // most diffuse textures use 0, TODO: more efficient manage

namespace KE {

Rasterizer::Rasterizer(Backend::GraphicsDevice* graphicsDevice) {
    _graphicsDevice = graphicsDevice;
    _cameraUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                               2 * sizeof(glm::mat4));
    _graphicsDevice->bindUniformBuffer(_cameraUBO.get(), CAMERA_UBO_BIND_SLOT);
    _lightUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                              3 * sizeof(glm::vec4));
    _graphicsDevice->bindUniformBuffer(_lightUBO.get(), LIGHT_UBO_BIND_SLOT);
    // std140 shadow data: mat4[4] cascade matrices, vec4 cascade splits,
    // vec4 cascade ortho half-sizes, vec4 cascade map sizes, vec4 params,
    // vec4 info.
    _shadowUBO = _graphicsDevice->createBuffer(
        Backend::BufferType::Uniform,
        MaxShadowCascades * sizeof(glm::mat4) + 5 * sizeof(glm::vec4));
    _graphicsDevice->bindUniformBuffer(_shadowUBO.get(), SHADOW_UBO_BIND_SLOT);

    _shadowFbo = _graphicsDevice->createFramebuffer(
        {_shadowMapWH, _shadowMapWH, true, false, 0}); // depth-only, no MSAA
    for (size_t i = 0; i < _cascadeFbos.size(); ++i) {
        const int mapSize = _cascadeMapSizes[i];
        auto& fbo = _cascadeFbos[i];
        fbo = _graphicsDevice->createFramebuffer(
            {mapSize, mapSize, true, false, 0});
    }
    _shadowShader = _graphicsDevice->createShaderFromFile(
        KE::getAssetPath("shaders/shadow.vs"),
        KE::getAssetPath("shaders/shadow.fs"));
    _skinnedShadowShader = _graphicsDevice->createShaderFromFile(
        KE::getAssetPath("shaders/skinned_shadow.vs"),
        KE::getAssetPath("shaders/shadow.fs"));
    _debugRenderer.init(_graphicsDevice);
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

MeshHandle
Rasterizer::addSkinnedShape(Backend::Shader* shader, Scene::Prim* prim,
                            const Scene::SkinnedMeshData& skinnedMesh,
                            RenderTrack track) {
    auto meshData = prim->resolveMeshData();
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty() ||
        !skinnedMesh.hasValidVertexSkinning())
        return InvalidHandle;

    InstancerKey key{shader, meshData.get(), nullptr, track};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, shader, skinnedMesh);
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

void Rasterizer::updateSkinningMatrices(
    MeshHandle handle, const std::vector<glm::mat4>& boneMatrices) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->updateSkinningMatrices(boneMatrices);
}

void Rasterizer::logDebugLines(const std::string& path,
                               const std::vector<glm::vec3>& starts,
                               const std::vector<glm::vec3>& ends,
                               const std::vector<glm::vec4>& colors,
                               float width, bool hidden) {
    _debugRenderer.logLines(path, starts, ends, colors, width, hidden);
}

void Rasterizer::logDebugAxes(const std::string& path,
                              const glm::mat4& transform, float length,
                              float width, bool hidden) {
    _debugRenderer.logAxes(path, transform, length, width, hidden);
}

void Rasterizer::logDebugAxes(const std::string& path, const glm::vec3& origin,
                              const glm::vec3& xAxis, const glm::vec3& yAxis,
                              const glm::vec3& zAxis, float length, float width,
                              bool hidden) {
    _debugRenderer.logAxes(path, origin, xAxis, yAxis, zAxis, length, width,
                           hidden);
}

void Rasterizer::clearDebugLines(const std::string& path) {
    _debugRenderer.clearLines(path);
}

void Rasterizer::logDebugPoints(const std::string& path,
                                const std::vector<glm::vec3>& points,
                                const std::vector<glm::vec4>& colors,
                                float size, bool hidden) {
    _debugRenderer.logPoints(path, points, colors, size, hidden);
}

void Rasterizer::clearDebugPoints(const std::string& path) {
    _debugRenderer.clearPoints(path);
}

// Render

void Rasterizer::updateFrameData(const glm::mat4& view, const glm::mat4& proj) {
    _viewFrustum = Geometry::Frustum::fromViewProjection(proj * view);

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
    _cullingTotalBatches = 0;
    _cullingCulledBatches = 0;
    _cullingTotalInstances = 0;
    _cullingCulledInstances = 0;

    // Bind shadow map and upload shadow uniforms onto each instancer's shader
    const bool hasShadow = (_shadowMap != nullptr && _shadowDistance > 0.0f);
    Backend::Texture* fallbackShadowTexture =
        _shadowFbo ? _shadowFbo->getDepthTexture() : nullptr;
    Backend::Texture* shadowTexture =
        hasShadow ? _shadowMap : fallbackShadowTexture;

    auto bindShadowTextures = [&]() {
        if (!shadowTexture)
            return;
        for (int i = 0; i < MaxShadowCascades; ++i) {
            Backend::Texture* cascadeTexture =
                (_useCsm && hasShadow && _cascadeMaps[static_cast<size_t>(i)])
                    ? _cascadeMaps[static_cast<size_t>(i)]
                    : shadowTexture;
            cascadeTexture->bind(SHADOW_TEXTURE_SLOT_BASE + i);
        }
    };
    bindShadowTextures();

    auto bindShadowSampler = [&](Backend::Shader* sh) {
        if (!shadowTexture)
            return;
        sh->setInt("shadowMap0", SHADOW_TEXTURE_SLOT_BASE);
        sh->setInt("shadowMap1", SHADOW_TEXTURE_SLOT_BASE + 1);
        sh->setInt("shadowMap2", SHADOW_TEXTURE_SLOT_BASE + 2);
        sh->setInt("shadowMap3", SHADOW_TEXTURE_SLOT_BASE + 3);
        sh->setInt("debugCsmCascadeTint",
                   (_debugCsmCascadeTint && _useCsm) ? 1 : 0);
    };

    // Pass 1: opaque instanced
    for (auto& [key, inst] : _instancers) {
        if (inst.hasTransparent() || inst.visibleCount() == 0)
            continue;
        if (_frustumCullingEnabled) {
            ++_cullingTotalBatches;
            const int totalInstances = inst.instanceCount();
            _cullingTotalInstances += totalInstances;
            inst.applyFrustumCulling(&_viewFrustum);
            const int culledInstances = totalInstances - inst.visibleCount();
            _cullingCulledInstances += culledInstances;
            if (inst.visibleCount() == 0) {
                ++_cullingCulledBatches;
                continue;
            }
        }
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
        inst.uploadSkinningMatrices();
        // Re-bind shadow map after bindTextures() to prevent slot 1 conflict
        bindShadowTextures();
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
        if (_frustumCullingEnabled) {
            ++_cullingTotalBatches;
            const int totalInstances = inst.instanceCount();
            _cullingTotalInstances += totalInstances;
            inst.applyFrustumCulling(&_viewFrustum);
            const int culledInstances = totalInstances - inst.visibleCount();
            _cullingCulledInstances += culledInstances;
            if (inst.visibleCount() == 0) {
                ++_cullingCulledBatches;
                continue;
            }
        }
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
        bindShadowTextures();
        inst.uploadSkinningMatrices();
        inst.render();
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(true);
    }
    _graphicsDevice->setDepthWrite(true);
    _graphicsDevice->setBlend(false);

    _debugRenderer.render();
}

/////////////// Shadow Pass //////////////

void Rasterizer::updateShadowUBO(float activeOrthoHalfSize) {
    std::array<glm::mat4, MaxShadowCascades> matrices{};
    std::array<float, MaxShadowCascades> orthoHalfSizes{};
    matrices.fill(glm::mat4(1.0f));

    const bool csmActive = _useCsm && _shadowMap && _shadowDistance > 0.0f;
    if (csmActive) {
        matrices = _cascadeLightMatrices;
        orthoHalfSizes = _cascadeOrthoHalfSizes;
    } else {
        matrices[0] = _lightSpaceMatrix;
        orthoHalfSizes[0] = activeOrthoHalfSize;
    }

    const glm::vec4 cascadeSplits{_cascadeSplits[0], _cascadeSplits[1],
                                  _cascadeSplits[2], _cascadeSplits[3]};
    const glm::vec4 cascadeOrthoHalfSizes{orthoHalfSizes[0], orthoHalfSizes[1],
                                          orthoHalfSizes[2], orthoHalfSizes[3]};
    const glm::vec4 cascadeMapSizes{
        static_cast<float>(csmActive ? _cascadeMapSizes[0] : _shadowMapWH),
        static_cast<float>(csmActive ? _cascadeMapSizes[1] : _shadowMapWH),
        static_cast<float>(csmActive ? _cascadeMapSizes[2] : _shadowMapWH),
        static_cast<float>(csmActive ? _cascadeMapSizes[3] : _shadowMapWH)};
    glm::vec4 shadowParams{_light.direction, _shadowRadius};
    glm::vec4 shadowInfo{static_cast<float>(_shadowPcfSamples),
                         csmActive ? static_cast<float>(_cascadeCount) : 0.0f,
                         csmActive ? 1.0f : 0.0f, 0.0f};

    size_t offset = 0;
    _shadowUBO->setData(matrices.data(), sizeof(glm::mat4) * matrices.size(),
                        offset);
    offset += sizeof(glm::mat4) * matrices.size();
    _shadowUBO->setData(&cascadeSplits, sizeof(cascadeSplits), offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&cascadeOrthoHalfSizes, sizeof(cascadeOrthoHalfSizes),
                        offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&cascadeMapSizes, sizeof(cascadeMapSizes), offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&shadowParams, sizeof(shadowParams), offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&shadowInfo, sizeof(shadowInfo), offset);
}

void Rasterizer::updateShadowPassUBO(const glm::mat4& lightSpaceMatrix,
                                     float activeOrthoHalfSize) {
    std::array<glm::mat4, MaxShadowCascades> matrices{};
    matrices.fill(glm::mat4(1.0f));
    matrices[0] = lightSpaceMatrix;

    const glm::vec4 cascadeSplits{0.0f};
    const glm::vec4 cascadeOrthoHalfSizes{activeOrthoHalfSize, 0.0f, 0.0f,
                                          0.0f};
    const glm::vec4 cascadeMapSizes{static_cast<float>(_shadowMapWH), 0.0f,
                                    0.0f, 0.0f};
    glm::vec4 shadowParams{_light.direction, _shadowRadius};
    glm::vec4 shadowInfo{static_cast<float>(_shadowPcfSamples), 0.0f, 0.0f,
                         0.0f};

    size_t offset = 0;
    _shadowUBO->setData(matrices.data(), sizeof(glm::mat4) * matrices.size(),
                        offset);
    offset += sizeof(glm::mat4) * matrices.size();
    _shadowUBO->setData(&cascadeSplits, sizeof(cascadeSplits), offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&cascadeOrthoHalfSizes, sizeof(cascadeOrthoHalfSizes),
                        offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&cascadeMapSizes, sizeof(cascadeMapSizes), offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&shadowParams, sizeof(shadowParams), offset);
    offset += sizeof(glm::vec4);
    _shadowUBO->setData(&shadowInfo, sizeof(shadowInfo), offset);
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
    const float shadowNear = camera.getNearPlane();
    const float shadowFar = std::min(camera.getFarPlane(), _shadowDistance);
    return computeLightSpaceMatrix(camera, upAxis, shadowNear, shadowFar);
}

std::array<float, Rasterizer::MaxShadowCascades>
Rasterizer::computeCascadeSplits(Camera& camera) {
    std::array<float, MaxShadowCascades> splits{};
    const float shadowNear = std::max(0.001f, camera.getNearPlane());
    const float shadowFar = std::max(
        shadowNear + 0.001f, std::min(camera.getFarPlane(), _shadowDistance));
    const float range = shadowFar - shadowNear;
    const float ratio = shadowFar / shadowNear;

    for (int i = 1; i <= _cascadeCount; ++i) {
        const float p =
            static_cast<float>(i) / static_cast<float>(_cascadeCount);
        const float logSplit = shadowNear * std::pow(ratio, p);
        const float uniformSplit = shadowNear + range * p;
        splits[i - 1] = glm::mix(uniformSplit, logSplit, _cascadeLambda);
    }
    return splits;
}

glm::mat4 Rasterizer::computeLightSpaceMatrix(Camera& camera,
                                              const UpAxis upAxis,
                                              float shadowNear,
                                              float shadowFar) {
    const glm::vec3& sunDir = glm::normalize(_light.direction);
    glm::vec3 up =
        (std::abs(sunDir.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    if (upAxis == UpAxis::Z)
        up = (std::abs(sunDir.z) < 0.99f) ? glm::vec3(0, 0, 1)
                                          : glm::vec3(1, 0, 0);

    shadowNear = std::max(0.001f, shadowNear);
    shadowFar = std::max(shadowNear + 0.001f, shadowFar);
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

    const float cascadeRange = shadowFar - shadowNear;
    glm::vec3 lightPos = frustumCenter + sunDir * std::max(10.0f, cascadeRange);
    glm::mat4 lightView = glm::lookAt(lightPos, frustumCenter, up);
    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();
    std::vector<glm::vec3> lightSpaceCorners;
    lightSpaceCorners.reserve(8);
    for (const auto& corner : corners) {
        glm::vec3 lightSpaceCorner =
            glm::vec3(lightView * glm::vec4(corner, 1.0f));
        lightSpaceCorners.push_back(lightSpaceCorner);
        minZ = std::min(minZ, lightSpaceCorner.z);
        maxZ = std::max(maxZ, lightSpaceCorner.z);
    }

    const float zPadding = std::max(5.0f, cascadeRange * 0.25f);
    glm::mat4 lightProj(1.0f);
    if (_useTightShadowFit) {
        const float casterMargin = std::max(1.0f, cascadeRange * 0.05f);
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();
        for (const auto& corner : lightSpaceCorners) {
            minX = std::min(minX, corner.x);
            maxX = std::max(maxX, corner.x);
            minY = std::min(minY, corner.y);
            maxY = std::max(maxY, corner.y);
        }

        minX -= casterMargin;
        maxX += casterMargin;
        minY -= casterMargin;
        maxY += casterMargin;
        _activeShadowOrthoHalfSize =
            std::max((maxX - minX) * 0.5f, (maxY - minY) * 0.5f);
        lightProj = glm::ortho(minX, maxX, minY, maxY, -maxZ - zPadding,
                               -minZ + zPadding);
    } else {
        float frustumRadius = 0.0f;
        for (const auto& corner : corners)
            frustumRadius =
                std::max(frustumRadius, glm::length(corner - frustumCenter));
        frustumRadius += std::max(2.0f, cascadeRange * 0.25f);
        _activeShadowOrthoHalfSize = frustumRadius;
        lightProj =
            glm::ortho(-frustumRadius, frustumRadius, -frustumRadius,
                       frustumRadius, -maxZ - zPadding, -minZ + zPadding);
    }

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

    _cascadeSplits = computeCascadeSplits(camera);
    float cascadeNear = camera.getNearPlane();
    for (int i = 0; i < _cascadeCount; ++i) {
        const float cascadeFar = _cascadeSplits[i];
        _cascadeLightMatrices[i] =
            computeLightSpaceMatrix(camera, upAxis, cascadeNear, cascadeFar);
        _cascadeOrthoHalfSizes[i] = _activeShadowOrthoHalfSize;
        cascadeNear = cascadeFar;
    }

    _lightSpaceMatrix = _useCsm ? _cascadeLightMatrices[0]
                                : computeLightSpaceMatrix(camera, upAxis);
    if (!_useCsm)
        _cascadeOrthoHalfSizes[0] = _activeShadowOrthoHalfSize;
    _shadowMap = _useCsm && _cascadeFbos[0] ? _cascadeFbos[0]->getDepthTexture()
                                            : _shadowFbo->getDepthTexture();
    updateShadowUBO(_shadowMap ? _activeShadowOrthoHalfSize : 0.0f);

    if (_useCsm) {
        for (int i = 0; i < _cascadeCount; ++i) {
            if (!_cascadeFbos[static_cast<size_t>(i)])
                continue;
            _lightSpaceMatrix = _cascadeLightMatrices[static_cast<size_t>(i)];
            _cascadeMaps[static_cast<size_t>(i)] =
                _cascadeFbos[static_cast<size_t>(i)]->getDepthTexture();
            updateShadowPassUBO(_cascadeLightMatrices[static_cast<size_t>(i)],
                                _cascadeOrthoHalfSizes[static_cast<size_t>(i)]);

            _cascadeFbos[static_cast<size_t>(i)]->bind();
            const int mapSize = _cascadeMapSizes[static_cast<size_t>(i)];
            _graphicsDevice->setViewport(0, 0, mapSize, mapSize);
            _graphicsDevice->clear(0, 0, 0, 1);
            drawShadowCasters();
            _cascadeFbos[static_cast<size_t>(i)]->unbind();
        }
        _lightSpaceMatrix = _cascadeLightMatrices[0];
        _shadowMap = _cascadeMaps[0];
        updateShadowUBO(_shadowMap ? _activeShadowOrthoHalfSize : 0.0f);
    } else {
        updateShadowPassUBO(_lightSpaceMatrix, _activeShadowOrthoHalfSize);
        _shadowFbo->bind();
        _graphicsDevice->setViewport(0, 0, _shadowMapWH, _shadowMapWH);
        _graphicsDevice->clear(0, 0, 0, 1);
        drawShadowCasters();
        _shadowFbo->unbind();
    }
    _graphicsDevice->setViewport(0, 0, viewportWidth, viewportHeight);
}

void Rasterizer::drawShadowCasters() {
    // Front-face culling avoids storing the same front surfaces that receive
    // the shadow, reducing acne on closed meshes.
    _graphicsDevice->setCullFaceMode(Backend::CullFaceMode::Front);
    for (auto& [key, inst] : _instancers) {
        if (inst.visibleCount() == 0)
            continue;
        if (!inst.castsShadow())
            continue;
        // Shadow pass runs after updateFrameData() uploads all instances and
        // before scene-pass frustum culling compacts the visible buffer.
        if (inst.hasSkinning() && _skinnedShadowShader) {
            _skinnedShadowShader->use();
            inst.uploadSkinningMatrices(_skinnedShadowShader.get());
        } else {
            _shadowShader->use();
        }
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(false);
        inst.render();
        if (inst.isDoubleSided())
            _graphicsDevice->setCullFace(true);
    }
    _graphicsDevice->setCullFaceMode(Backend::CullFaceMode::Back);
}

} // namespace KE

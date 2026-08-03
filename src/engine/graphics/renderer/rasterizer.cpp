#include "rasterizer.hpp"
#include <stdexcept>
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

// Depth/selection passes with no fragment material only need a trivial output.
// Production vertex and alpha-mask fragment sources live in assets/shaders.
constexpr const char* OpaqueMaskRhiFs = R"(
#version 410 core
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(1.0); }
)";
constexpr const char* DepthOnlyRhiFs = R"(
#version 410 core
void main() {}
)";

constexpr size_t shadowPipelineIndex(bool skin, bool alpha, bool doubleSided) {
    return (skin ? 4u : 0u) | (alpha ? 2u : 0u) |
           (doubleSided ? 1u : 0u);
}
constexpr int LIGHT_UBO_BIND_SLOT = 1;
constexpr int SHADOW_UBO_BIND_SLOT = 2;
constexpr int SHADOW_TEXTURE_SLOT_BASE = KE::RendererTextureSlot::Shadow0;
constexpr size_t LIGHT_UBO_VEC4_COUNT =
    3 + KE::MaxPointLights * 2 + KE::MaxSpotLights * 3 + 1;

namespace {

void appendAABBLines(const KE::Geometry::AABB& box,
                     std::vector<glm::vec3>& starts,
                     std::vector<glm::vec3>& ends,
                     std::vector<glm::vec4>& colors, const glm::vec4& color) {
    if (!box.isValid())
        return;

    const glm::vec3 mn = box.min;
    const glm::vec3 mx = box.max;
    const glm::vec3 corners[] = {
        {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mx.x, mx.y, mn.z},
        {mn.x, mx.y, mn.z}, {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
        {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
    };
    constexpr int edges[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
        {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (const auto& edge : edges) {
        starts.push_back(corners[edge[0]]);
        ends.push_back(corners[edge[1]]);
        colors.push_back(color);
    }
}

} // namespace

namespace KE {

Rasterizer::Rasterizer(Backend::GraphicsDevice* graphicsDevice) {
    _graphicsDevice = graphicsDevice;
    _cameraUBO = _graphicsDevice->createBuffer(Backend::BufferType::Uniform,
                                               2 * sizeof(glm::mat4));
    _graphicsDevice->bindUniformBuffer(_cameraUBO.get(), CAMERA_UBO_BIND_SLOT);
    _lightUBO = _graphicsDevice->createBuffer(
        Backend::BufferType::Uniform, LIGHT_UBO_VEC4_COUNT * sizeof(glm::vec4));
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
    initSelectionMaskRhi();
    initShadowRhi();
    initForwardRhi();
    initSkyboxRhi();
    _debugRenderer.init(_graphicsDevice, _forwardGroupLayouts[0].get(),
                        _forwardFrameBindGroup.get());
    _textRenderer.init(
        _graphicsDevice,
        FontAtlasData::loadAscii(KE::getAssetPath("fonts/godoFont/GodoM.ttf")),
        _forwardGroupLayouts[0].get(), _forwardFrameBindGroup.get());
    updateShadowUBO(0.0f);
}

// Prim-based (instanced)

void Rasterizer::registerPrimSource(Scene::Prim* prim, TransformSource source) {
    if (!prim)
        return;
    PrimSourceRegistrations& registrations = _primSourceRegistrations[prim];
    const bool sourceConflict = source == TransformSource::ExternalBuffer
                                    ? registrations.sceneGraph > 0
                                    : registrations.external > 0;
    if (sourceConflict) {
        const char* existingSource =
            registrations.external > 0 ? "ExternalBuffer" : "SceneGraph";
        const char* requestedSource = source == TransformSource::ExternalBuffer
                                          ? "ExternalBuffer"
                                          : "SceneGraph";
        throw std::runtime_error(
            "Prim '" + prim->getPath() + "' is already registered with " +
            existingSource + "; cannot also register it with " +
            requestedSource);
    }
    if (source == TransformSource::ExternalBuffer)
        ++registrations.external;
    else
        ++registrations.sceneGraph;
}

void Rasterizer::unregisterPrimSource(Scene::Prim* prim,
                                      TransformSource source) {
    auto it = _primSourceRegistrations.find(prim);
    if (it == _primSourceRegistrations.end())
        return;
    uint32_t& count = source == TransformSource::ExternalBuffer
                          ? it->second.external
                          : it->second.sceneGraph;
    if (count > 0)
        --count;
    if (it->second.external == 0 && it->second.sceneGraph == 0)
        _primSourceRegistrations.erase(it);
}

RenderableHandle
Rasterizer::addSkinnedRenderable(Material* material, Scene::Prim* prim,
                                 const Scene::SkinnedMeshData& skinnedMesh,
                                 TransformSource transformSource) {
    auto meshData = prim->resolveMeshData();
    if (!material || !meshData || meshData->vertices.empty() ||
        meshData->indices.empty() || !skinnedMesh.hasValidVertexSkinning())
        return InvalidHandle;

    auto* shader = material->getShader();
    InstancerKey key{shader, meshData.get(), material, transformSource};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, shader, skinnedMesh,
                           transformSource, material);
        it = newIt;
    }
    registerPrimSource(prim, transformSource);
    it->second.addPrim(prim);

    auto hIt = _handleMap.find(key);
    if (hIt == _handleMap.end()) {
        RenderableHandle h = static_cast<RenderableHandle>(_handleTable.size());
        _handleMap[key] = h;
        _handleTable.push_back(&it->second);
        return h;
    }
    return hIt->second;
}

RenderableHandle Rasterizer::addRenderable(Material* material,
                                           Scene::Prim* prim,
                                           TransformSource transformSource) {
    auto meshData = prim->resolveMeshData();
    if (!material || !meshData || meshData->vertices.empty() ||
        meshData->indices.empty())
        return InvalidHandle;

    auto* shader = material->getShader();
    InstancerKey key{shader, meshData.get(), material, transformSource};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, shader, *meshData, transformSource,
                           material);
        it = newIt;
    }
    registerPrimSource(prim, transformSource);
    it->second.addPrim(prim);

    auto hIt = _handleMap.find(key);
    if (hIt == _handleMap.end()) {
        RenderableHandle h = static_cast<RenderableHandle>(_handleTable.size());
        _handleMap[key] = h;
        _handleTable.push_back(&it->second);
        return h;
    }
    return hIt->second;
}

void Rasterizer::removePrim(RenderableHandle handle, Scene::Prim* prim) {
    if (handle >= _handleTable.size() || !_handleTable[handle])
        return;
    unregisterPrimSource(prim, _handleTable[handle]->transformSource());
    _handleTable[handle]->removePrim(prim);
}

void Rasterizer::removePrim(Scene::Prim* prim) {
    if (!prim)
        return;
    for (auto& [key, instancer] : _instancers)
        instancer.removePrim(prim);
    _primSourceRegistrations.erase(prim);
}

void Rasterizer::updateRenderableTransforms(
    RenderableHandle handle, const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->updateFromTransforms(transforms, colors);
}

void Rasterizer::setRenderableExternalBuffer(RenderableHandle handle,
                                             const ExternalBufferDesc& desc) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setExternalBuffer(desc);
}

std::vector<Sim::GpuArrayView> Rasterizer::mapRenderableCudaTransformBuffers(
    const std::vector<RenderableHandle>& handles, int count, int deviceId,
    uint64_t streamHandle) {
    std::vector<Backend::Buffer*> buffers;
    buffers.reserve(handles.size());
    for (auto handle : handles) {
        if (handle >= _handleTable.size())
            throw std::out_of_range("invalid renderable handle for CUDA map");
        auto* instancer = _handleTable[handle];
        instancer->prepareDirectCudaTransforms(count);
        buffers.push_back(instancer->transformBuffer());
    }
    std::vector<Sim::GpuArrayView> views;
    if (!_graphicsDevice->mapCudaBuffers(
            buffers, views, static_cast<size_t>(count), sizeof(glm::mat4),
            deviceId, streamHandle))
        throw std::runtime_error(
            "graphics backend cannot map CUDA transform buffers");
    return views;
}

void Rasterizer::unmapRenderableCudaTransformBuffers(
    const std::vector<RenderableHandle>& handles, int deviceId,
    uint64_t streamHandle) {
    std::vector<Backend::Buffer*> buffers;
    buffers.reserve(handles.size());
    for (auto handle : handles) {
        if (handle >= _handleTable.size())
            throw std::out_of_range("invalid renderable handle for CUDA unmap");
        buffers.push_back(_handleTable[handle]->transformBuffer());
    }
    _graphicsDevice->unmapCudaBuffers(buffers, deviceId, streamHandle);
}

void Rasterizer::setRenderableColors(RenderableHandle handle,
                                     const std::vector<glm::vec4>& colors) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setColors(colors);
}

void Rasterizer::setRenderableDoubleSided(RenderableHandle handle,
                                          bool doubleSided) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setDoubleSided(doubleSided);
}

void Rasterizer::setRenderableCastsShadow(RenderableHandle handle,
                                          bool castsShadow) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setCastsShadow(castsShadow);
}

void Rasterizer::setRenderableAlphaMode(RenderableHandle handle, AlphaMode mode,
                                        float cutoff) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setAlphaMode(mode, cutoff);
}

void Rasterizer::setRenderableTexture(RenderableHandle handle,
                                      Backend::Texture* tex, TextureRole role) {
    setRenderableTexture(handle, tex, textureRoleSlot(role));
}

void Rasterizer::setRenderableTexture(RenderableHandle handle,
                                      Backend::Texture* tex, int slot) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->setTexture(tex, slot);
}

RayPickResult Rasterizer::rayPick(const Geometry::Ray& ray) const {
    RayPickResult best;
    best.distance = std::numeric_limits<float>::infinity();
    for (size_t handle = 0; handle < _handleTable.size(); ++handle) {
        const MeshInstancer* inst = _handleTable[handle];
        if (!inst)
            continue;

        int instanceIndex = -1;
        float distance = 0.0f;
        Geometry::AABB bounds;
        Scene::Prim* prim = nullptr;
        if (!inst->findRayIntersection(ray, instanceIndex, distance, &bounds,
                                       &prim))
            continue;
        if (distance >= best.distance)
            continue;

        best.hit = true;
        best.handle = static_cast<RenderableHandle>(handle);
        best.instanceIndex = instanceIndex;
        best.transformSource = inst->transformSource();
        best.prim = best.transformSource == TransformSource::SceneGraph
                        ? prim
                        : nullptr;
        best.distance = distance;
        best.position = ray.getPoint(distance);
        best.bounds = bounds;
    }
    return best;
}

bool Rasterizer::buildPrimSelection(Scene::Prim* prim,
                                    RayPickResult& outSelection) const {
    outSelection = RayPickResult{};
    for (size_t handle = 0; handle < _handleTable.size(); ++handle) {
        const MeshInstancer* inst = _handleTable[handle];
        if (!inst || inst->transformSource() != TransformSource::SceneGraph)
            continue;

        int instanceIndex = -1;
        Geometry::AABB bounds;
        if (!inst->findPrimInstance(prim, instanceIndex, &bounds))
            continue;

        outSelection.hit = true;
        outSelection.handle = static_cast<RenderableHandle>(handle);
        outSelection.instanceIndex = instanceIndex;
        outSelection.transformSource = TransformSource::SceneGraph;
        outSelection.prim = prim;
        outSelection.bounds = bounds;
        return true;
    }
    return false;
}

bool Rasterizer::getPrimTransformSource(const Scene::Prim* prim,
                                        TransformSource& outSource) const {
    const auto it = _primSourceRegistrations.find(prim);
    if (it == _primSourceRegistrations.end())
        return false;
    outSource = it->second.external > 0 ? TransformSource::ExternalBuffer
                                        : TransformSource::SceneGraph;
    return true;
}

bool Rasterizer::getRenderableInstanceTransform(RenderableHandle handle,
                                                int instanceIndex,
                                                glm::mat4& outTransform) const {
    if (handle >= _handleTable.size() || !_handleTable[handle])
        return false;
    return _handleTable[handle]->getInstanceTransform(instanceIndex,
                                                      outTransform);
}

bool Rasterizer::setRenderableInstanceTransform(RenderableHandle handle,
                                                int instanceIndex,
                                                const glm::mat4& transform) {
    if (handle >= _handleTable.size() || !_handleTable[handle])
        return false;

    MeshInstancer* instancer = _handleTable[handle];
    if (instancer->transformSource() != TransformSource::ExternalBuffer)
        return false;

    return instancer->setInstanceTransform(instanceIndex, transform);
}

void Rasterizer::updateRenderableGeometry(
    RenderableHandle handle, const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->updateGeometry(positions, normals);
}

void Rasterizer::updateRenderableSkinningMatrices(
    RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices) {
    if (handle >= _handleTable.size())
        return;
    _handleTable[handle]->updateRenderableSkinningMatrices(boneMatrices);
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

void Rasterizer::setWorldText(const std::string& path,
                              const WorldTextDesc& desc) {
    _textRenderer.setWorldText(path, desc);
}

void Rasterizer::setWorldTextString(const std::string& path,
                                    std::string text) {
    _textRenderer.setWorldString(path, std::move(text));
}

void Rasterizer::setWorldTextPosition(const std::string& path,
                                      const glm::vec3& position) {
    _textRenderer.setWorldPosition(path, position);
}

void Rasterizer::setWorldTextHidden(const std::string& path, bool hidden) {
    _textRenderer.setWorldHidden(path, hidden);
}

void Rasterizer::removeWorldText(const std::string& path) {
    _textRenderer.removeWorldText(path);
}

void Rasterizer::clearWorldText() { _textRenderer.clearWorldText(); }

void Rasterizer::setScreenText(const std::string& path,
                               const ScreenTextDesc& desc) {
    _textRenderer.setScreenText(path, desc);
}

void Rasterizer::setScreenTextString(const std::string& path,
                                     std::string text) {
    _textRenderer.setScreenString(path, std::move(text));
}

void Rasterizer::setScreenTextPosition(const std::string& path,
                                       const glm::vec2& position) {
    _textRenderer.setScreenPosition(path, position);
}

void Rasterizer::setScreenTextHidden(const std::string& path, bool hidden) {
    _textRenderer.setScreenHidden(path, hidden);
}

void Rasterizer::removeScreenText(const std::string& path) {
    _textRenderer.removeScreenText(path);
}

void Rasterizer::clearScreenText() { _textRenderer.clearScreenText(); }

void Rasterizer::updateDebugRenderAABB() {
    constexpr const char* path = "/renderer/aabb";
    if (!_debugRenderAABB) {
        _debugRenderer.clearLines(path);
        return;
    }

    std::vector<glm::vec3> starts;
    std::vector<glm::vec3> ends;
    std::vector<glm::vec4> colors;
    const glm::vec4 color(0.45f, 1.0f, 0.55f, 1.0f);

    for (const auto& [key, inst] : _instancers) {
        if (inst.visibleCount() == 0)
            continue;

        const auto& boundsList = inst.worldBounds();
        starts.reserve(starts.size() + boundsList.size() * 12);
        ends.reserve(ends.size() + boundsList.size() * 12);
        colors.reserve(colors.size() + boundsList.size() * 12);
        for (const auto& bounds : boundsList) {
            if (_frustumCullingEnabled &&
                !Geometry::intersects(_viewFrustum, bounds))
                continue;
            appendAABBLines(bounds, starts, ends, colors, color);
        }
    }

    _debugRenderer.logLines(path, starts, ends, colors, 1.0f, starts.empty());
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

    // std140 layout keeps legacy fields first:
    // vec4 directional dir | vec4 directional color | vec4 ambient
    // vec4 point position+range[4] | vec4 point color+intensity[4]
    // vec4 spot position+range[2] | vec4 spot direction+innerCos[2]
    // vec4 spot color+outerCos[2] | ivec4 counts
    glm::vec4 lightColor = glm::vec4(_light.color * _light.intensity, 1.f);
    glm::vec4 ambient = glm::vec4(_light.ambient, 0.f);
    _lightUBO->setData(&lightColor, sizeof(glm::vec4), 1 * sizeof(glm::vec4));
    _lightUBO->setData(&ambient, sizeof(glm::vec4), 2 * sizeof(glm::vec4));

    std::array<glm::vec4, MaxPointLights> pointPositionRange{};
    std::array<glm::vec4, MaxPointLights> pointColorIntensity{};
    const size_t pointCount =
        std::min(_pointLights.size(), static_cast<size_t>(MaxPointLights));
    for (size_t i = 0; i < pointCount; ++i) {
        const PointLight& light = _pointLights[i];
        const glm::vec3 viewPos =
            glm::vec3(view * glm::vec4(light.position, 1.0f));
        pointPositionRange[i] =
            glm::vec4(viewPos, std::max(light.range, 0.0001f));
        pointColorIntensity[i] = glm::vec4(light.color * light.intensity, 1.0f);
    }

    std::array<glm::vec4, MaxSpotLights> spotPositionRange{};
    std::array<glm::vec4, MaxSpotLights> spotDirectionInner{};
    std::array<glm::vec4, MaxSpotLights> spotColorOuter{};
    const size_t spotCount =
        std::min(_spotLights.size(), static_cast<size_t>(MaxSpotLights));
    for (size_t i = 0; i < spotCount; ++i) {
        const SpotLight& light = _spotLights[i];
        const glm::vec3 viewPos =
            glm::vec3(view * glm::vec4(light.position, 1.0f));
        glm::vec3 direction = light.direction;
        if (glm::length(direction) <= 0.0001f)
            direction = glm::vec3(0.0f, 0.0f, -1.0f);
        const glm::vec3 viewDir = glm::normalize(glm::mat3(view) * direction);
        const float innerAngle =
            std::min(light.innerConeAngle, light.outerConeAngle);
        const float outerAngle =
            std::max(light.innerConeAngle, light.outerConeAngle);
        spotPositionRange[i] =
            glm::vec4(viewPos, std::max(light.range, 0.0001f));
        spotDirectionInner[i] = glm::vec4(viewDir, std::cos(innerAngle));
        spotColorOuter[i] =
            glm::vec4(light.color * light.intensity, std::cos(outerAngle));
    }

    constexpr size_t pointPositionOffset = 3;
    constexpr size_t pointColorOffset = pointPositionOffset + MaxPointLights;
    constexpr size_t spotPositionOffset = pointColorOffset + MaxPointLights;
    constexpr size_t spotDirectionOffset = spotPositionOffset + MaxSpotLights;
    constexpr size_t spotColorOffset = spotDirectionOffset + MaxSpotLights;
    constexpr size_t countOffset = spotColorOffset + MaxSpotLights;

    _lightUBO->setData(pointPositionRange.data(), sizeof(pointPositionRange),
                       pointPositionOffset * sizeof(glm::vec4));
    _lightUBO->setData(pointColorIntensity.data(), sizeof(pointColorIntensity),
                       pointColorOffset * sizeof(glm::vec4));
    _lightUBO->setData(spotPositionRange.data(), sizeof(spotPositionRange),
                       spotPositionOffset * sizeof(glm::vec4));
    _lightUBO->setData(spotDirectionInner.data(), sizeof(spotDirectionInner),
                       spotDirectionOffset * sizeof(glm::vec4));
    _lightUBO->setData(spotColorOuter.data(), sizeof(spotColorOuter),
                       spotColorOffset * sizeof(glm::vec4));
    glm::ivec4 lightCounts(static_cast<int>(pointCount),
                           static_cast<int>(spotCount), 0, 0);
    _lightUBO->setData(&lightCounts, sizeof(lightCounts),
                       countOffset * sizeof(glm::vec4));
    _lightDirty = false;

    // Update all instancers (must run before shadow pass AND scene pass)
    for (auto& [key, inst] : _instancers)
        inst.update();
}

void Rasterizer::render(const glm::mat4& view, const glm::mat4& proj,
                        Backend::RenderTarget* sceneDrawTarget) {
    _cullingTotalBatches = 0;
    _cullingCulledBatches = 0;
    _cullingTotalInstances = 0;
    _cullingCulledInstances = 0;

    Backend::Texture* shadowTexture = activeShadowTexture();
    rebuildShadowSamplingBindings(shadowTexture);
    bindShadowTextures(shadowTexture);
    renderOpaquePass(shadowTexture, sceneDrawTarget);
    recordRenderHooks(RenderHookPhase::AfterOpaque, sceneDrawTarget);
    renderSkyboxPass(view, proj, sceneDrawTarget);
    renderTransparentPass(shadowTexture, sceneDrawTarget);
    recordRenderHooks(RenderHookPhase::AfterTransparent, sceneDrawTarget);
    renderDebugOverlayPass(sceneDrawTarget);
    _textRenderer.render(sceneDrawTarget, _viewportWidth, _viewportHeight);
}

RenderHookHandle Rasterizer::addRenderHook(RenderHookPhase phase,
                                           RenderHookCallback callback) {
    if (!callback)
        throw std::invalid_argument("render hook callback is empty");
    const size_t phaseIndex = static_cast<size_t>(phase);
    if (phaseIndex >= _renderHooks.size())
        throw std::invalid_argument("render hook phase is invalid");
    const RenderHookHandle handle = _nextRenderHook++;
    _renderHooks[phaseIndex].push_back({handle, std::move(callback)});
    return handle;
}

bool Rasterizer::removeRenderHook(RenderHookHandle handle) {
    if (handle == InvalidRenderHook)
        return false;
    for (auto& hooks : _renderHooks) {
        const auto found = std::find_if(
            hooks.begin(), hooks.end(),
            [handle](const RenderHookEntry& entry) {
                return entry.handle == handle;
            });
        if (found != hooks.end()) {
            hooks.erase(found);
            return true;
        }
    }
    return false;
}

void Rasterizer::recordRenderHooks(
    RenderHookPhase phase, Backend::RenderTarget* sceneDrawTarget) {
    if (!sceneDrawTarget)
        return;
    const size_t phaseIndex = static_cast<size_t>(phase);
    if (phaseIndex >= _renderHooks.size() || _renderHooks[phaseIndex].empty())
        return;

    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(sceneDrawTarget);
    RenderHookContext context{*pass, *sceneDrawTarget,
                              _forwardFrameBindGroup.get(), _viewportWidth,
                              _viewportHeight};
    for (const auto& hook : _renderHooks[phaseIndex])
        hook.callback(context);
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

Backend::Texture* Rasterizer::activeShadowTexture() const {
    const bool hasShadow = (_shadowMap != nullptr && _shadowDistance > 0.0f);
    Backend::Texture* fallbackShadowTexture =
        _shadowFbo ? _shadowFbo->getDepthTexture() : nullptr;
    return hasShadow ? _shadowMap : fallbackShadowTexture;
}

void Rasterizer::bindShadowTextures(Backend::Texture* shadowTexture) {
    if (!shadowTexture)
        return;

    const bool hasShadow = (_shadowMap != nullptr && _shadowDistance > 0.0f);
    for (int i = 0; i < MaxShadowCascades; ++i) {
        Backend::Texture* cascadeTexture =
            (_useCsm && hasShadow && _cascadeMaps[static_cast<size_t>(i)])
                ? _cascadeMaps[static_cast<size_t>(i)]
                : shadowTexture;
        cascadeTexture->bind(SHADOW_TEXTURE_SLOT_BASE + i);
    }
}

// -------------------------------------------------------------------------
// Opaque Scene Pipeline
// -------------------------------------------------------------------------
void Rasterizer::renderOpaquePass(Backend::Texture* shadowTexture,
                                  Backend::RenderTarget* sceneDrawTarget) {
    if (sceneDrawTarget && _forwardFrameBindGroup &&
        _shadowSamplingBindGroup) {
        auto encoder = _graphicsDevice->createCommandEncoder();
        auto pass = encoder->beginRenderPass(sceneDrawTarget);
        std::vector<std::unique_ptr<Backend::BindGroup>> transientSkinGroups;
        pass->setViewport(0.0f, 0.0f, static_cast<float>(_viewportWidth),
                          static_cast<float>(_viewportHeight));
        for (auto& entry : _instancers) {
            MeshInstancer& inst = entry.second;
            if (inst.hasTransparent() || inst.visibleCount() == 0)
                continue;
            if (_frustumCullingEnabled) {
                ++_cullingTotalBatches;
                const int total = inst.instanceCount();
                _cullingTotalInstances += total;
                inst.applyFrustumCulling(&_viewFrustum);
                const int culled = total - inst.visibleCount();
                _cullingCulledInstances += culled;
                if (inst.visibleCount() == 0) {
                    ++_cullingCulledBatches;
                    continue;
                }
            }
            const bool texturedVertexColor =
                usesRhiTexturedVertexColor(inst);
            const bool checkerboard = usesRhiCheckerboard(inst);
            const bool skinnedTexturedVertexColor =
                inst.hasSkinning() && texturedVertexColor;
            const bool skinnedVertexColor =
                inst.hasSkinning() && inst.material()->shadingModel() ==
                                          MaterialShadingModel::VertexColor &&
                !texturedVertexColor;
            const bool skinnedMaterial =
                inst.hasSkinning() &&
                (inst.material()->shadingModel() == MaterialShadingModel::Phong ||
                 inst.material()->shadingModel() == MaterialShadingModel::PBR);
            Backend::GraphicsPipeline* pipeline = nullptr;
            if (checkerboard)
                pipeline = _checkerboardPipelines[0]
                    [inst.isDoubleSided() ? 1 : 0].get();
            else if (texturedVertexColor)
                pipeline = _texturedVertexColorPipelines
                    [inst.hasSkinning() ? 1 : 0][0]
                    [inst.isDoubleSided() ? 1 : 0].get();
            else if (skinnedVertexColor)
                pipeline = inst.isDoubleSided()
                               ? _forwardSkinDoubleSidedPipeline.get()
                               : _forwardSkinPipeline.get();
            else if (skinnedMaterial) {
                const size_t model = inst.material()->shadingModel() ==
                                             MaterialShadingModel::Phong
                                         ? 0 : 1;
                pipeline = _skinnedMaterialPipelines[model][0]
                    [inst.isDoubleSided() ? 1 : 0].get();
            }
            else if (inst.material()->shadingModel() ==
                     MaterialShadingModel::Phong)
                pipeline = inst.isDoubleSided()
                               ? _phongDoubleSidedPipeline.get()
                               : _phongPipeline.get();
            else if (inst.material()->shadingModel() ==
                     MaterialShadingModel::PBR)
                pipeline = inst.isDoubleSided()
                               ? _pbrDoubleSidedPipeline.get()
                               : _pbrPipeline.get();
            else
                pipeline = inst.isDoubleSided()
                               ? _forwardDoubleSidedPipeline.get()
                               : _forwardPipeline.get();
            pass->setPipeline(pipeline);
            pass->setBindGroup(0, _forwardFrameBindGroup.get());
            pass->setBindGroup(1, _shadowSamplingBindGroup.get());
            if (skinnedVertexColor || skinnedTexturedVertexColor ||
                skinnedMaterial) {
                Backend::BindGroupDesc skinDesc;
                skinDesc.layout = _forwardSkinGroupLayout.get();
                skinDesc.label = "forward_skin_bind_group";
                skinDesc.entries = {{0, inst.boneMatricesBuffer(), 0,
                                     inst.boneMatricesBuffer()->getSize(),
                                     nullptr, nullptr}};
                transientSkinGroups.push_back(
                    _graphicsDevice->createBindGroup(skinDesc));
                pass->setBindGroup(2, transientSkinGroups.back().get());
                if (skinnedTexturedVertexColor) {
                    pass->setBindGroup(
                        3, updateTexturedVertexColorRhiResources(inst));
                    inst.recordSkinnedMaterialDraw(
                        *pass,
                        inst.textureAtSlot(RendererTextureSlot::Normal) !=
                            nullptr);
                } else if (skinnedVertexColor) {
                    inst.recordSkinnedForwardDraw(*pass);
                } else if (inst.material()->shadingModel() ==
                           MaterialShadingModel::Phong) {
                    auto* material = static_cast<PhongMaterial*>(inst.material());
                    pass->setBindGroup(3,
                        updatePhongRhiResources(*material, inst));
                    inst.recordSkinnedMaterialDraw(
                        *pass, material->normalMap != nullptr);
                } else {
                    auto* material = static_cast<PBRMaterial*>(inst.material());
                    pass->setBindGroup(3,
                        updatePbrRhiResources(*material, inst));
                    inst.recordSkinnedMaterialDraw(
                        *pass, material->normalTexture != nullptr);
                }
            } else if (checkerboard) {
                pass->setBindGroup(3, _checkerboardBindGroup.get());
                inst.recordMaterialDraw(*pass, true, false);
            } else if (texturedVertexColor) {
                pass->setBindGroup(
                    3, updateTexturedVertexColorRhiResources(inst));
                inst.recordMaterialDraw(
                    *pass, true,
                    inst.textureAtSlot(RendererTextureSlot::Normal) != nullptr);
            } else if (inst.material()->shadingModel() ==
                MaterialShadingModel::Phong) {
                auto* phong = static_cast<PhongMaterial*>(inst.material());
                pass->setBindGroup(3,
                                   updatePhongRhiResources(*phong, inst));
                inst.recordMaterialDraw(*pass, true, phong->normalMap != nullptr);
            } else if (inst.material()->shadingModel() ==
                       MaterialShadingModel::PBR) {
                auto* pbr = static_cast<PBRMaterial*>(inst.material());
                pass->setBindGroup(3, updatePbrRhiResources(*pbr, inst));
                inst.recordMaterialDraw(*pass, true,
                                        pbr->normalTexture != nullptr);
            } else {
                inst.recordForwardDraw(*pass);
            }
        }
        pass->end();
        auto commands = encoder->finish();
        _graphicsDevice->submit(*commands);
    }
}

// -------------------------------------------------------------------------
// Skybox Pipeline
// -------------------------------------------------------------------------
void Rasterizer::renderSkyboxPass(const glm::mat4& view,
                                  const glm::mat4& proj,
                                  Backend::RenderTarget* sceneDrawTarget) {
    // Drawn after opaque geometry so the skybox only fills empty pixels.
    if (!sceneDrawTarget || !_skyboxPipeline || !_skyboxTextureBindGroup) {
        return;
    }
    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(sceneDrawTarget);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(_viewportWidth),
                      static_cast<float>(_viewportHeight));
    pass->setPipeline(_skyboxPipeline.get());
    pass->setBindGroup(0, _forwardFrameBindGroup.get());
    pass->setBindGroup(1, _skyboxParamsBindGroup.get());
    pass->setBindGroup(3, _skyboxTextureBindGroup.get());
    pass->setVertexBuffer(0, _skyboxVertexBuffer.get());
    pass->setIndexBuffer(_skyboxIndexBuffer.get(), Backend::IndexFormat::Uint32);
    pass->drawIndexed(36);
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

// -------------------------------------------------------------------------
// Transparent Scene Pipeline
// -------------------------------------------------------------------------
void Rasterizer::renderTransparentPass(
    Backend::Texture* shadowTexture, Backend::RenderTarget* sceneDrawTarget) {
    if (sceneDrawTarget && _shadowSamplingBindGroup) {
        auto encoder = _graphicsDevice->createCommandEncoder();
        auto pass = encoder->beginRenderPass(sceneDrawTarget);
        pass->setViewport(0.0f, 0.0f, static_cast<float>(_viewportWidth),
                          static_cast<float>(_viewportHeight));
        std::vector<std::unique_ptr<Backend::BindGroup>> skinGroups;
        for (auto& entry : _instancers) {
            MeshInstancer& inst = entry.second;
            if (!inst.hasTransparent() || inst.visibleCount() == 0)
                continue;
            size_t kind = 0;
            const bool texturedVertexColor =
                usesRhiTexturedVertexColor(inst);
            const bool checkerboard = usesRhiCheckerboard(inst);
            const bool skinnedMaterial = inst.hasSkinning() &&
                (inst.material()->shadingModel() == MaterialShadingModel::Phong ||
                 inst.material()->shadingModel() == MaterialShadingModel::PBR);
            if (inst.hasSkinning() && !skinnedMaterial) kind = 1;
            else if (inst.material()->shadingModel() == MaterialShadingModel::Phong) kind = 2;
            else if (inst.material()->shadingModel() == MaterialShadingModel::PBR) kind = 3;
            if (checkerboard) {
                pass->setPipeline(_checkerboardPipelines[1]
                    [inst.isDoubleSided() ? 1 : 0].get());
            } else if (texturedVertexColor) {
                pass->setPipeline(_texturedVertexColorPipelines
                    [inst.hasSkinning() ? 1 : 0][1]
                    [inst.isDoubleSided() ? 1 : 0].get());
            } else if (skinnedMaterial) {
                const size_t model = inst.material()->shadingModel() ==
                                             MaterialShadingModel::Phong
                                         ? 0 : 1;
                pass->setPipeline(_skinnedMaterialPipelines[model][1]
                    [inst.isDoubleSided() ? 1 : 0].get());
            } else {
                pass->setPipeline(_transparentPipelines[kind]
                    [inst.isDoubleSided() ? 1 : 0].get());
            }
            pass->setBindGroup(0, _forwardFrameBindGroup.get());
            pass->setBindGroup(1, _shadowSamplingBindGroup.get());
            if (kind == 1 || skinnedMaterial) {
                Backend::BindGroupDesc desc;
                desc.layout = _forwardSkinGroupLayout.get();
                desc.label = "transparent_skin_bind_group";
                desc.entries = {{0, inst.boneMatricesBuffer(), 0,
                                 inst.boneMatricesBuffer()->getSize(), nullptr,
                                 nullptr}};
                skinGroups.push_back(_graphicsDevice->createBindGroup(desc));
                pass->setBindGroup(2, skinGroups.back().get());
                if (texturedVertexColor) {
                    pass->setBindGroup(
                        3, updateTexturedVertexColorRhiResources(inst));
                    inst.recordSkinnedMaterialDraw(
                        *pass,
                        inst.textureAtSlot(RendererTextureSlot::Normal) !=
                            nullptr);
                } else if (!skinnedMaterial) {
                    inst.recordSkinnedForwardDraw(*pass);
                } else if (inst.material()->shadingModel() ==
                           MaterialShadingModel::Phong) {
                    auto* material = static_cast<PhongMaterial*>(inst.material());
                    pass->setBindGroup(3,
                        updatePhongRhiResources(*material, inst));
                    inst.recordSkinnedMaterialDraw(
                        *pass, material->normalMap != nullptr);
                } else {
                    auto* material = static_cast<PBRMaterial*>(inst.material());
                    pass->setBindGroup(3,
                        updatePbrRhiResources(*material, inst));
                    inst.recordSkinnedMaterialDraw(
                        *pass, material->normalTexture != nullptr);
                }
            } else if (checkerboard) {
                pass->setBindGroup(3, _checkerboardBindGroup.get());
                inst.recordMaterialDraw(*pass, true, false);
            } else if (texturedVertexColor) {
                pass->setBindGroup(
                    3, updateTexturedVertexColorRhiResources(inst));
                inst.recordMaterialDraw(
                    *pass, true,
                    inst.textureAtSlot(RendererTextureSlot::Normal) != nullptr);
            } else if (kind == 2) {
                auto* material = static_cast<PhongMaterial*>(inst.material());
                pass->setBindGroup(3, updatePhongRhiResources(*material, inst));
                inst.recordMaterialDraw(*pass, true, material->normalMap != nullptr);
            } else if (kind == 3) {
                auto* material = static_cast<PBRMaterial*>(inst.material());
                pass->setBindGroup(3, updatePbrRhiResources(*material, inst));
                inst.recordMaterialDraw(*pass, true, material->normalTexture != nullptr);
            } else {
                inst.recordForwardDraw(*pass);
            }
        }
        pass->end();
        auto commands = encoder->finish();
        _graphicsDevice->submit(*commands);
    }
}

// -------------------------------------------------------------------------
// Debug Overlay Pipeline
// -------------------------------------------------------------------------
void Rasterizer::renderDebugOverlayPass(
    Backend::RenderTarget* sceneDrawTarget) {
    updateDebugRenderAABB();
    _debugRenderer.render(sceneDrawTarget, _viewportWidth, _viewportHeight);
}

void Rasterizer::initSelectionMaskRhi() {
    // =====================================================================
    // Selection Mask Pipeline
    // Produces the binary mask consumed by SelectionOutlineProcessor.
    // Variants: opaque/alpha-mask, static/skinned, culled/double-sided.
    // =====================================================================
    const std::string selectionVertexSource = Backend::loadShaderSource(
        KE::getAssetPath("shaders/selection_mask.vs"));
    const std::string selectionFragmentSource = Backend::loadShaderSource(
        KE::getAssetPath("shaders/selection_mask.fs"));
    const std::string skinnedSelectionVertexSource =
        Backend::loadShaderSource(
            KE::getAssetPath("shaders/skinned_selection_mask.vs"));

    // Selection pipeline layouts and frame bindings.
    Backend::BindGroupLayoutDesc frameDesc;
    frameDesc.label = "selection_mask_frame_layout";
    frameDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                          Backend::ShaderStageVisibility::Vertex}};
    _selectionMaskGroupLayouts[0] =
        _graphicsDevice->createBindGroupLayout(frameDesc);
    for (size_t i = 1; i < _selectionMaskGroupLayouts.size(); ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "selection_mask_empty_group_" + std::to_string(i);
        _selectionMaskGroupLayouts[i] =
            _graphicsDevice->createBindGroupLayout(emptyDesc);
    }
    Backend::PipelineLayoutDesc layoutDesc;
    layoutDesc.label = "selection_mask_pipeline_layout";
    for (const auto& layout : _selectionMaskGroupLayouts)
        layoutDesc.bindGroupLayouts.push_back(layout.get());
    _selectionMaskPipelineLayout =
        _graphicsDevice->createPipelineLayout(layoutDesc);

    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "selection_mask_opaque_pipeline";
    pipelineDesc.shader.name = "selection_mask_opaque_rhi";
    pipelineDesc.shader.stages = {
        {selectionVertexSource, Backend::ShaderType::Vertex, "main"},
        {OpaqueMaskRhiFs, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.pipelineLayout = _selectionMaskPipelineLayout.get();
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA8Unorm}};
    Backend::VertexBufferLayout positionLayout;
    positionLayout.arrayStride = sizeof(glm::vec3);
    positionLayout.attributes = {{Backend::VertexFormat::Float32x3, 0,
                                  RendererAttribute::Position}};
    Backend::VertexBufferLayout transformLayout;
    transformLayout.arrayStride = sizeof(glm::mat4);
    transformLayout.stepMode = Backend::VertexStepMode::Instance;
    for (uint32_t column = 0; column < 4; ++column) {
        transformLayout.attributes.push_back(
            {Backend::VertexFormat::Float32x4,
             column * sizeof(glm::vec4),
             static_cast<uint32_t>(RendererAttribute::InstanceTransform0) +
                 column});
    }
    pipelineDesc.vertexBuffers = {positionLayout, transformLayout};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    _selectionMaskPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "selection_mask_opaque_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _selectionMaskDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    // Alpha-mask selection pipeline: UV + sampled alpha + cutoff parameters.
    Backend::BindGroupLayoutDesc alphaLayoutDesc;
    alphaLayoutDesc.label = "selection_mask_alpha_group_layout";
    alphaLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {1, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _selectionMaskAlphaGroupLayout =
        _graphicsDevice->createBindGroupLayout(alphaLayoutDesc);
    Backend::PipelineLayoutDesc alphaPipelineLayoutDesc;
    alphaPipelineLayoutDesc.label = "selection_mask_alpha_pipeline_layout";
    for (size_t i = 0; i < 3; ++i)
        alphaPipelineLayoutDesc.bindGroupLayouts.push_back(
            _selectionMaskGroupLayouts[i].get());
    alphaPipelineLayoutDesc.bindGroupLayouts.push_back(
        _selectionMaskAlphaGroupLayout.get());
    _selectionMaskAlphaPipelineLayout =
        _graphicsDevice->createPipelineLayout(alphaPipelineLayoutDesc);

    pipelineDesc.label = "selection_mask_alpha_pipeline";
    pipelineDesc.shader.name = "selection_mask_alpha_rhi";
    pipelineDesc.shader.stages = {
        {selectionVertexSource, Backend::ShaderType::Vertex, "main"},
        {selectionFragmentSource, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.pipelineLayout = _selectionMaskAlphaPipelineLayout.get();
    Backend::VertexBufferLayout emptyLayout;
    Backend::VertexBufferLayout texCoordLayout;
    texCoordLayout.arrayStride = sizeof(glm::vec2);
    texCoordLayout.attributes = {{Backend::VertexFormat::Float32x2, 0,
                                  RendererAttribute::TexCoord}};
    pipelineDesc.vertexBuffers = {positionLayout, transformLayout, emptyLayout,
                                  texCoordLayout};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    _selectionMaskAlphaPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "selection_mask_alpha_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _selectionMaskAlphaDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    // Skinned selection pipelines: group 2 owns the bone-matrix UBO.
    Backend::BindGroupLayoutDesc skinLayoutDesc;
    skinLayoutDesc.label = "selection_mask_skin_group_layout";
    skinLayoutDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                               Backend::ShaderStageVisibility::Vertex}};
    _selectionMaskSkinGroupLayout =
        _graphicsDevice->createBindGroupLayout(skinLayoutDesc);
    auto makeSkinPipelineLayout = [&](bool alpha) {
        Backend::PipelineLayoutDesc desc;
        desc.label = alpha ? "selection_mask_skin_alpha_pipeline_layout"
                           : "selection_mask_skin_pipeline_layout";
        desc.bindGroupLayouts = {
            _selectionMaskGroupLayouts[0].get(),
            _selectionMaskGroupLayouts[1].get(),
            _selectionMaskSkinGroupLayout.get(),
            alpha ? _selectionMaskAlphaGroupLayout.get()
                  : _selectionMaskGroupLayouts[3].get(),
        };
        return _graphicsDevice->createPipelineLayout(desc);
    };
    _selectionMaskSkinPipelineLayout = makeSkinPipelineLayout(false);
    _selectionMaskSkinAlphaPipelineLayout = makeSkinPipelineLayout(true);

    Backend::VertexBufferLayout boneIndexLayout;
    boneIndexLayout.arrayStride = sizeof(glm::ivec4);
    boneIndexLayout.attributes = {{Backend::VertexFormat::Sint32x4, 0,
                                   RendererAttribute::BoneIndices}};
    Backend::VertexBufferLayout boneWeightLayout;
    boneWeightLayout.arrayStride = sizeof(glm::vec4);
    boneWeightLayout.attributes = {{Backend::VertexFormat::Float32x4, 0,
                                    RendererAttribute::BoneWeights}};
    pipelineDesc.shader.name = "selection_mask_skin_rhi";
    pipelineDesc.shader.stages = {
        {skinnedSelectionVertexSource, Backend::ShaderType::Vertex, "main"},
        {OpaqueMaskRhiFs, Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.vertexBuffers = {
        positionLayout, transformLayout, emptyLayout, emptyLayout,
        emptyLayout,    emptyLayout,     boneIndexLayout, boneWeightLayout,
    };
    pipelineDesc.pipelineLayout = _selectionMaskSkinPipelineLayout.get();
    pipelineDesc.label = "selection_mask_skin_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    _selectionMaskSkinPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "selection_mask_skin_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _selectionMaskSkinDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    pipelineDesc.shader.name = "selection_mask_skin_alpha_rhi";
    pipelineDesc.shader.stages[1] =
        {selectionFragmentSource, Backend::ShaderType::Fragment, "main"};
    pipelineDesc.vertexBuffers[vertexBufferSlot(
        MeshVertexBufferSlot::TexCoord)] = texCoordLayout;
    pipelineDesc.pipelineLayout = _selectionMaskSkinAlphaPipelineLayout.get();
    pipelineDesc.label = "selection_mask_skin_alpha_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    _selectionMaskSkinAlphaPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "selection_mask_skin_alpha_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _selectionMaskSkinAlphaDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    // Shared immutable alpha sampler and per-frame camera binding.
    Backend::SamplerDesc alphaSamplerDesc;
    alphaSamplerDesc.wrapU = Backend::TextureWrap::Repeat;
    alphaSamplerDesc.wrapV = Backend::TextureWrap::Repeat;
    alphaSamplerDesc.minFilter = Backend::TextureFilter::Linear;
    alphaSamplerDesc.magFilter = Backend::TextureFilter::Linear;
    alphaSamplerDesc.label = "selection_mask_alpha_sampler";
    _selectionMaskAlphaSampler =
        _graphicsDevice->createSampler(alphaSamplerDesc);

    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _selectionMaskGroupLayouts[0].get();
    bindDesc.label = "selection_mask_frame_bind_group";
    bindDesc.entries = {{0, _cameraUBO.get(), 0, 2 * sizeof(glm::mat4),
                         nullptr, nullptr}};
    _selectionMaskFrameBindGroup =
        _graphicsDevice->createBindGroup(bindDesc);
}

void Rasterizer::initShadowRhi() {
    // =====================================================================
    // Shadow Depth Pipeline
    // Records single-map and CSM depth passes. The 8 immutable variants cover
    // static/skinned, opaque/alpha-mask, and culled/double-sided casters.
    // =====================================================================
    const std::string shadowVertexSource = Backend::loadShaderSource(
        KE::getAssetPath("shaders/shadow.vs"));
    const std::string shadowFragmentSource = Backend::loadShaderSource(
        KE::getAssetPath("shaders/shadow.fs"));
    const std::string skinnedShadowVertexSource = Backend::loadShaderSource(
        KE::getAssetPath("shaders/skinned_shadow.vs"));

    // Shadow frame binding and mesh vertex layouts.
    Backend::BindGroupDesc frameBindDesc;
    frameBindDesc.layout = _selectionMaskGroupLayouts[0].get();
    frameBindDesc.label = "shadow_frame_bind_group";
    frameBindDesc.entries = {{
        0, _shadowUBO.get(), 0,
        MaxShadowCascades * sizeof(glm::mat4) + 5 * sizeof(glm::vec4), nullptr,
        nullptr}};
    _shadowFrameBindGroup = _graphicsDevice->createBindGroup(frameBindDesc);

    Backend::VertexBufferLayout positionLayout;
    positionLayout.arrayStride = sizeof(glm::vec3);
    positionLayout.attributes = {{Backend::VertexFormat::Float32x3, 0,
                                  RendererAttribute::Position}};
    Backend::VertexBufferLayout transformLayout;
    transformLayout.arrayStride = sizeof(glm::mat4);
    transformLayout.stepMode = Backend::VertexStepMode::Instance;
    for (uint32_t column = 0; column < 4; ++column) {
        transformLayout.attributes.push_back(
            {Backend::VertexFormat::Float32x4,
             column * sizeof(glm::vec4),
             static_cast<uint32_t>(RendererAttribute::InstanceTransform0) +
                 column});
    }
    Backend::VertexBufferLayout texCoordLayout;
    texCoordLayout.arrayStride = sizeof(glm::vec2);
    texCoordLayout.attributes = {{Backend::VertexFormat::Float32x2, 0,
                                  RendererAttribute::TexCoord}};
    Backend::VertexBufferLayout boneIndexLayout;
    boneIndexLayout.arrayStride = sizeof(glm::ivec4);
    boneIndexLayout.attributes = {{Backend::VertexFormat::Sint32x4, 0,
                                   RendererAttribute::BoneIndices}};
    Backend::VertexBufferLayout boneWeightLayout;
    boneWeightLayout.arrayStride = sizeof(glm::vec4);
    boneWeightLayout.attributes = {{Backend::VertexFormat::Float32x4, 0,
                                    RendererAttribute::BoneWeights}};
    Backend::VertexBufferLayout emptyLayout;

    // Immutable shadow pipeline variants.
    for (bool skin : {false, true}) {
        for (bool alpha : {false, true}) {
            for (bool doubleSided : {false, true}) {
                Backend::GraphicsPipelineDesc desc;
                desc.label = "shadow_rhi_pipeline";
                desc.shader.name = "shadow_rhi";
                desc.shader.stages = {
                    {skin ? skinnedShadowVertexSource : shadowVertexSource,
                     Backend::ShaderType::Vertex, "main"},
                    {alpha ? shadowFragmentSource
                           : std::string(DepthOnlyRhiFs),
                     Backend::ShaderType::Fragment, "main"},
                };
                desc.vertexBuffers = {
                    positionLayout, transformLayout, emptyLayout,
                    alpha ? texCoordLayout : emptyLayout, emptyLayout,
                    emptyLayout, skin ? boneIndexLayout : emptyLayout,
                    skin ? boneWeightLayout : emptyLayout,
                };
                if (skin && alpha)
                    desc.pipelineLayout =
                        _selectionMaskSkinAlphaPipelineLayout.get();
                else if (skin)
                    desc.pipelineLayout =
                        _selectionMaskSkinPipelineLayout.get();
                else if (alpha)
                    desc.pipelineLayout =
                        _selectionMaskAlphaPipelineLayout.get();
                else
                    desc.pipelineLayout = _selectionMaskPipelineLayout.get();
                desc.primitive.cullMode = doubleSided
                                              ? Backend::CullMode::None
                                              : Backend::CullMode::Back;
                desc.depthStencil = Backend::DepthStencilState{
                    Backend::TextureFormat::Depth32Float, true,
                    Backend::CompareFunction::Less,
                    1,    // constant bias: one implementation depth unit
                    1.0f, // slope bias: reduces acne on grazing surfaces
                    0.0f};
                _shadowPipelines[shadowPipelineIndex(skin, alpha,
                                                     doubleSided)] =
                    _graphicsDevice->createGraphicsPipeline(desc);
            }
        }
    }

    // Depth-only render targets for the single shadow map and CSM cascades.
    auto createDepthTarget = [&](Backend::Framebuffer* framebuffer,
                                 const std::string& label,
                                 std::unique_ptr<Backend::TextureView>& view,
                                 std::unique_ptr<Backend::RenderTarget>& target) {
        Backend::Texture* depth =
            framebuffer ? framebuffer->getDepthTexture() : nullptr;
        if (!depth)
            return;
        Backend::TextureViewDesc viewDesc;
        viewDesc.format = depth->getFormat();
        viewDesc.aspect = Backend::TextureAspect::DepthOnly;
        viewDesc.label = label + "_view";
        view = _graphicsDevice->createTextureView(depth, viewDesc);
        Backend::RenderPassDesc passDesc;
        passDesc.label = label;
        passDesc.depthStencilAttachment = Backend::DepthStencilAttachmentDesc{
            view.get(), Backend::LoadOp::Clear, Backend::StoreOp::Store, 1.0f,
            Backend::LoadOp::Load, Backend::StoreOp::Store, 0};
        target = _graphicsDevice->createRenderTarget(passDesc);
    };
    createDepthTarget(_shadowFbo.get(), "shadow_depth_pass", _shadowDepthView,
                      _shadowRenderTarget);
    for (size_t i = 0; i < _cascadeFbos.size(); ++i) {
        createDepthTarget(_cascadeFbos[i].get(),
                          "shadow_cascade_pass_" + std::to_string(i),
                          _cascadeDepthViews[i], _cascadeRenderTargets[i]);
    }

    // Shadow Sampling Bind Group
    // This is kept separate from shadow-map production. The forward RHI scene
    // pipeline will consume the four CSM textures through this single group.
    Backend::BindGroupLayoutDesc samplingLayoutDesc;
    samplingLayoutDesc.label = "shadow_sampling_group_layout";
    for (uint32_t binding = 0; binding < MaxShadowCascades; ++binding) {
        samplingLayoutDesc.entries.push_back(
            {binding, Backend::BindingType::SampledTexture,
             Backend::ShaderStageVisibility::Fragment,
             Backend::TextureFormat::Depth32Float,
             Backend::TextureSampleType::Depth});
    }
    samplingLayoutDesc.entries.push_back(
        {4, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment});
    samplingLayoutDesc.entries.push_back(
        {5, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment});
    _shadowSamplingGroupLayout =
        _graphicsDevice->createBindGroupLayout(samplingLayoutDesc);

    Backend::SamplerDesc shadowSamplerDesc;
    shadowSamplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    shadowSamplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    shadowSamplerDesc.minFilter = Backend::TextureFilter::Nearest;
    shadowSamplerDesc.magFilter = Backend::TextureFilter::Nearest;
    shadowSamplerDesc.label = "shadow_sampling_sampler";
    _shadowSamplingSampler =
        _graphicsDevice->createSampler(shadowSamplerDesc);

    Backend::BufferDesc samplingParamsDesc;
    samplingParamsDesc.size = sizeof(glm::vec4);
    samplingParamsDesc.usage = Backend::BufferUsage::Uniform |
                               Backend::BufferUsage::CopyDst;
    samplingParamsDesc.label = "shadow_sampling_params";
    _shadowSamplingParamsBuffer =
        _graphicsDevice->createBuffer(samplingParamsDesc);
    rebuildShadowSamplingBindings(activeShadowTexture());
}

void Rasterizer::rebuildShadowSamplingBindings(
    Backend::Texture* fallbackTexture) {
    if (!fallbackTexture || !_shadowSamplingGroupLayout)
        return;

    const bool hasShadow = _shadowMap && _shadowDistance > 0.0f;
    std::array<Backend::Texture*, MaxShadowCascades> textures{};
    std::array<uintptr_t, MaxShadowCascades> handles{};
    for (size_t i = 0; i < textures.size(); ++i) {
        textures[i] = (_useCsm && hasShadow && _cascadeMaps[i])
                          ? _cascadeMaps[i]
                          : fallbackTexture;
        handles[i] = textures[i]->getNativeHandle();
    }

    const glm::vec4 params{
        (_debugCsmCascadeTint && _useCsm) ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f};
    _shadowSamplingParamsBuffer->setData(&params, sizeof(params));
    if (_shadowSamplingBindGroup && handles == _shadowSamplingHandles)
        return;

    _shadowSamplingBindGroup.reset();
    for (size_t i = 0; i < textures.size(); ++i) {
        Backend::TextureViewDesc viewDesc;
        viewDesc.format = Backend::TextureFormat::Depth32Float;
        viewDesc.aspect = Backend::TextureAspect::DepthOnly;
        viewDesc.label = "shadow_sampling_view_" + std::to_string(i);
        _shadowSamplingViews[i] =
            _graphicsDevice->createTextureView(textures[i], viewDesc);
    }

    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _shadowSamplingGroupLayout.get();
    bindDesc.label = "shadow_sampling_bind_group";
    for (uint32_t binding = 0; binding < MaxShadowCascades; ++binding) {
        bindDesc.entries.push_back(
            {binding, nullptr, 0, 0,
             _shadowSamplingViews[binding].get(), nullptr});
    }
    bindDesc.entries.push_back(
        {4, nullptr, 0, 0, nullptr, _shadowSamplingSampler.get()});
    bindDesc.entries.push_back(
        {5, _shadowSamplingParamsBuffer.get(), 0, sizeof(glm::vec4), nullptr,
         nullptr});
    _shadowSamplingBindGroup = _graphicsDevice->createBindGroup(bindDesc);
    _shadowSamplingHandles = handles;
}

void Rasterizer::initForwardRhi() {
    // Group 0 owns frame data. Group 1 reuses the shadow-sampling layout;
    // groups 2/3 stay reserved for skinning and material resources.
    Backend::BindGroupLayoutDesc frameDesc;
    frameDesc.label = "forward_frame_group_layout";
    frameDesc.entries = {
        {0, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Vertex},
        {1, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _forwardGroupLayouts[0] =
        _graphicsDevice->createBindGroupLayout(frameDesc);
    for (size_t i = 1; i < _forwardGroupLayouts.size(); ++i) {
        Backend::BindGroupLayoutDesc emptyDesc;
        emptyDesc.label = "forward_reserved_group_" + std::to_string(i + 1);
        _forwardGroupLayouts[i] =
            _graphicsDevice->createBindGroupLayout(emptyDesc);
    }

    Backend::PipelineLayoutDesc layoutDesc;
    layoutDesc.label = "forward_vertex_color_pipeline_layout";
    layoutDesc.bindGroupLayouts = {
        _forwardGroupLayouts[0].get(), _shadowSamplingGroupLayout.get(),
        _forwardGroupLayouts[1].get(), _forwardGroupLayouts[2].get()};
    _forwardPipelineLayout =
        _graphicsDevice->createPipelineLayout(layoutDesc);

    Backend::BindGroupDesc bindDesc;
    bindDesc.layout = _forwardGroupLayouts[0].get();
    bindDesc.label = "forward_frame_bind_group";
    bindDesc.entries = {
        {0, _cameraUBO.get(), 0, _cameraUBO->getSize(), nullptr, nullptr},
        {1, _lightUBO.get(), 0, _lightUBO->getSize(), nullptr, nullptr},
        {2, _shadowUBO.get(), 0, _shadowUBO->getSize(), nullptr, nullptr},
    };
    _forwardFrameBindGroup = _graphicsDevice->createBindGroup(bindDesc);

    Backend::VertexBufferLayout position;
    position.arrayStride = sizeof(glm::vec3);
    position.attributes = {{Backend::VertexFormat::Float32x3, 0,
                            RendererAttribute::Position}};
    Backend::VertexBufferLayout transform;
    transform.arrayStride = sizeof(glm::mat4);
    transform.stepMode = Backend::VertexStepMode::Instance;
    for (uint32_t column = 0; column < 4; ++column)
        transform.attributes.push_back(
            {Backend::VertexFormat::Float32x4, column * sizeof(glm::vec4),
             static_cast<uint32_t>(RendererAttribute::InstanceTransform0) +
                 column});
    Backend::VertexBufferLayout normal;
    normal.arrayStride = sizeof(glm::vec3);
    normal.attributes = {{Backend::VertexFormat::Float32x3, 0,
                          RendererAttribute::Normal}};
    Backend::VertexBufferLayout color;
    color.arrayStride = sizeof(glm::vec4);
    color.stepMode = Backend::VertexStepMode::Instance;
    color.attributes = {{Backend::VertexFormat::Float32x4, 0,
                         RendererAttribute::InstanceColor}};
    Backend::VertexBufferLayout empty;

    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "forward_vertex_color_pipeline";
    pipelineDesc.shader.name = "forward_vertex_color_rhi";
    pipelineDesc.shader.stages = {
        {Backend::loadShaderSource(
             KE::getAssetPath("shaders/rhi/forward_vertex_color.vs")),
         Backend::ShaderType::Vertex, "main"},
        {Backend::loadShaderSource(
             KE::getAssetPath("shaders/rhi/forward_vertex_color.fs")),
         Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.pipelineLayout = _forwardPipelineLayout.get();
    pipelineDesc.vertexBuffers = {position, transform, normal, empty, color};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    pipelineDesc.depthStencil = Backend::DepthStencilState{
        Backend::TextureFormat::Depth24Stencil8, true,
        Backend::CompareFunction::Less};
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.sampleCount = 4;
    _forwardPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "forward_vertex_color_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _forwardDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    Backend::BindGroupLayoutDesc skinLayoutDesc;
    skinLayoutDesc.label = "forward_skin_group_layout";
    skinLayoutDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                               Backend::ShaderStageVisibility::Vertex}};
    _forwardSkinGroupLayout =
        _graphicsDevice->createBindGroupLayout(skinLayoutDesc);
    Backend::PipelineLayoutDesc skinPipelineLayoutDesc;
    skinPipelineLayoutDesc.label = "forward_skin_pipeline_layout";
    skinPipelineLayoutDesc.bindGroupLayouts = {
        _forwardGroupLayouts[0].get(), _shadowSamplingGroupLayout.get(),
        _forwardSkinGroupLayout.get(), _forwardGroupLayouts[2].get()};
    _forwardSkinPipelineLayout =
        _graphicsDevice->createPipelineLayout(skinPipelineLayoutDesc);
    Backend::VertexBufferLayout boneIndices;
    boneIndices.arrayStride = sizeof(glm::ivec4);
    boneIndices.attributes = {{Backend::VertexFormat::Sint32x4, 0,
                               RendererAttribute::BoneIndices}};
    Backend::VertexBufferLayout boneWeights;
    boneWeights.arrayStride = sizeof(glm::vec4);
    boneWeights.attributes = {{Backend::VertexFormat::Float32x4, 0,
                               RendererAttribute::BoneWeights}};
    pipelineDesc.label = "forward_skinned_vertex_color_pipeline";
    pipelineDesc.shader.name = "forward_skinned_vertex_color_rhi";
    pipelineDesc.shader.stages[0] =
        {Backend::loadShaderSource(KE::getAssetPath(
             "shaders/rhi/forward_skinned_vertex_color.vs")),
         Backend::ShaderType::Vertex, "main"};
    pipelineDesc.pipelineLayout = _forwardSkinPipelineLayout.get();
    pipelineDesc.vertexBuffers = {
        position, transform, normal, empty, color, empty, boneIndices,
        boneWeights};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    _forwardSkinPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "forward_skinned_vertex_color_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _forwardSkinDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    // Phong Material Pipeline
    Backend::BindGroupLayoutDesc materialLayoutDesc;
    materialLayoutDesc.label = "phong_material_group_layout";
    materialLayoutDesc.entries = {
        {0, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
        {1, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {3, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {4, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {5, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
    };
    _phongMaterialGroupLayout =
        _graphicsDevice->createBindGroupLayout(materialLayoutDesc);
    Backend::PipelineLayoutDesc phongLayoutDesc;
    phongLayoutDesc.label = "forward_phong_pipeline_layout";
    phongLayoutDesc.bindGroupLayouts = {
        _forwardGroupLayouts[0].get(), _shadowSamplingGroupLayout.get(),
        _forwardGroupLayouts[1].get(), _phongMaterialGroupLayout.get()};
    _phongPipelineLayout =
        _graphicsDevice->createPipelineLayout(phongLayoutDesc);

    Backend::SamplerDesc materialSamplerDesc;
    materialSamplerDesc.wrapU = Backend::TextureWrap::Repeat;
    materialSamplerDesc.wrapV = Backend::TextureWrap::Repeat;
    materialSamplerDesc.minFilter = Backend::TextureFilter::Linear;
    materialSamplerDesc.magFilter = Backend::TextureFilter::Linear;
    materialSamplerDesc.label = "forward_material_sampler";
    _materialSampler = _graphicsDevice->createSampler(materialSamplerDesc);
    auto makeFallbackTexture = [&](const std::array<uint8_t, 4>& pixel,
                                   const char* label) {
        Backend::TextureResourceDesc desc;
        desc.extent = {1, 1, 1};
        desc.format = Backend::TextureFormat::RGBA8Unorm;
        desc.usage = Backend::TextureUsage::TextureBinding |
                     Backend::TextureUsage::CopyDst;
        desc.label = label;
        Backend::TextureInitialData initial{pixel.data(), pixel.size(), 4};
        return _graphicsDevice->createTexture(desc, &initial);
    };
    _materialWhiteTexture =
        makeFallbackTexture({255, 255, 255, 255}, "material_fallback_white");
    _materialNormalTexture =
        makeFallbackTexture({128, 128, 255, 255}, "material_fallback_normal");

    Backend::VertexBufferLayout texCoord;
    texCoord.arrayStride = sizeof(glm::vec2);
    texCoord.attributes = {{Backend::VertexFormat::Float32x2, 0,
                            RendererAttribute::TexCoord}};
    Backend::VertexBufferLayout tangent;
    tangent.arrayStride = sizeof(glm::vec4);
    tangent.attributes = {{Backend::VertexFormat::Float32x4, 0,
                           RendererAttribute::Tangent}};

    // Textured Vertex-Color Pipeline
    // Covers commonTex.fs users such as SkinVisualBridge and deformable cloth
    // without routing them through the legacy texture-slot path.
    Backend::BindGroupLayoutDesc texturedLayoutDesc;
    texturedLayoutDesc.label = "textured_vertex_color_group_layout";
    texturedLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {1, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment},
        {2, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment},
        {3, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment},
    };
    _texturedVertexColorGroupLayout =
        _graphicsDevice->createBindGroupLayout(texturedLayoutDesc);
    for (size_t skin = 0; skin < 2; ++skin) {
        Backend::PipelineLayoutDesc texturedPipelineLayoutDesc;
        texturedPipelineLayoutDesc.label = skin
            ? "forward_skinned_textured_vertex_color_pipeline_layout"
            : "forward_textured_vertex_color_pipeline_layout";
        texturedPipelineLayoutDesc.bindGroupLayouts = {
            _forwardGroupLayouts[0].get(), _shadowSamplingGroupLayout.get(),
            skin ? _forwardSkinGroupLayout.get() : _forwardGroupLayouts[1].get(),
            _texturedVertexColorGroupLayout.get()};
        _texturedVertexColorPipelineLayouts[skin] =
            _graphicsDevice->createPipelineLayout(texturedPipelineLayoutDesc);
    }
    Backend::BlendState texturedAlphaBlend;
    texturedAlphaBlend.color.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    texturedAlphaBlend.color.dstFactor =
        Backend::BlendFactorValue::OneMinusSrcAlpha;
    texturedAlphaBlend.alpha.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    texturedAlphaBlend.alpha.dstFactor =
        Backend::BlendFactorValue::OneMinusSrcAlpha;
    const std::string texturedFs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_textured_vertex_color.fs"));
    for (size_t skin = 0; skin < 2; ++skin) {
        const std::string texturedVs = Backend::loadShaderSource(KE::getAssetPath(
            skin ? "shaders/rhi/forward_skinned_material.vs"
                 : "shaders/rhi/forward_material.vs"));
        for (size_t transparent = 0; transparent < 2; ++transparent) {
            for (size_t doubleSided = 0; doubleSided < 2; ++doubleSided) {
                Backend::GraphicsPipelineDesc desc;
                desc.label = std::string("forward_") +
                    (skin ? "skinned_" : "") +
                    "textured_vertex_color_" +
                    (transparent ? "transparent_" : "opaque_") +
                    (doubleSided ? "double_sided" : "back_face");
                desc.shader.name = desc.label + "_rhi";
                desc.shader.stages = {
                    {texturedVs, Backend::ShaderType::Vertex, "main"},
                    {texturedFs, Backend::ShaderType::Fragment, "main"}};
                desc.pipelineLayout =
                    _texturedVertexColorPipelineLayouts[skin].get();
                desc.vertexBuffers = {position, transform, normal, texCoord,
                                      color, tangent};
                if (skin) {
                    desc.vertexBuffers.push_back(boneIndices);
                    desc.vertexBuffers.push_back(boneWeights);
                }
                desc.primitive.cullMode = doubleSided
                                              ? Backend::CullMode::None
                                              : Backend::CullMode::Back;
                desc.depthStencil = Backend::DepthStencilState{
                    Backend::TextureFormat::Depth24Stencil8, !transparent,
                    Backend::CompareFunction::Less};
                desc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
                if (transparent)
                    desc.colorTargets[0].blend = texturedAlphaBlend;
                desc.sampleCount = 4;
                _texturedVertexColorPipelines[skin][transparent][doubleSided] =
                    _graphicsDevice->createGraphicsPipeline(desc);
            }
        }
    }

    // Checkerboard Ground Pipeline
    Backend::BindGroupLayoutDesc checkerLayoutDesc;
    checkerLayoutDesc.label = "checkerboard_group_layout";
    checkerLayoutDesc.entries = {{
        0, Backend::BindingType::UniformBuffer,
        Backend::ShaderStageVisibility::Fragment}};
    _checkerboardGroupLayout =
        _graphicsDevice->createBindGroupLayout(checkerLayoutDesc);
    Backend::PipelineLayoutDesc checkerPipelineLayoutDesc;
    checkerPipelineLayoutDesc.label = "forward_checkerboard_pipeline_layout";
    checkerPipelineLayoutDesc.bindGroupLayouts = {
        _forwardGroupLayouts[0].get(), _shadowSamplingGroupLayout.get(),
        _forwardGroupLayouts[1].get(), _checkerboardGroupLayout.get()};
    _checkerboardPipelineLayout =
        _graphicsDevice->createPipelineLayout(checkerPipelineLayoutDesc);
    Backend::BufferDesc checkerParamsDesc;
    checkerParamsDesc.size = sizeof(glm::vec4) * 4;
    checkerParamsDesc.usage = Backend::BufferUsage::Uniform |
                              Backend::BufferUsage::CopyDst;
    checkerParamsDesc.label = "checkerboard_params";
    _checkerboardParamsBuffer =
        _graphicsDevice->createBuffer(checkerParamsDesc);
    Backend::BindGroupDesc checkerGroupDesc;
    checkerGroupDesc.layout = _checkerboardGroupLayout.get();
    checkerGroupDesc.label = "checkerboard_bind_group";
    checkerGroupDesc.entries = {{
        0, _checkerboardParamsBuffer.get(), 0, checkerParamsDesc.size, nullptr,
        nullptr}};
    _checkerboardBindGroup =
        _graphicsDevice->createBindGroup(checkerGroupDesc);
    setBackgroundSettings(BackgroundSettings{});

    const std::string checkerVs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_material.vs"));
    const std::string checkerFs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_checkerboard.fs"));
    for (size_t transparent = 0; transparent < 2; ++transparent) {
        for (size_t doubleSided = 0; doubleSided < 2; ++doubleSided) {
            Backend::GraphicsPipelineDesc desc;
            desc.label = std::string("forward_checkerboard_") +
                (transparent ? "transparent_" : "opaque_") +
                (doubleSided ? "double_sided" : "back_face");
            desc.shader.name = desc.label + "_rhi";
            desc.shader.stages = {
                {checkerVs, Backend::ShaderType::Vertex, "main"},
                {checkerFs, Backend::ShaderType::Fragment, "main"}};
            desc.pipelineLayout = _checkerboardPipelineLayout.get();
            desc.vertexBuffers = {position, transform, normal, texCoord, color,
                                  tangent};
            desc.primitive.cullMode = doubleSided
                                          ? Backend::CullMode::None
                                          : Backend::CullMode::Back;
            desc.depthStencil = Backend::DepthStencilState{
                Backend::TextureFormat::Depth24Stencil8, !transparent,
                Backend::CompareFunction::Less};
            desc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
            if (transparent)
                desc.colorTargets[0].blend = texturedAlphaBlend;
            desc.sampleCount = 4;
            _checkerboardPipelines[transparent][doubleSided] =
                _graphicsDevice->createGraphicsPipeline(desc);
        }
    }

    pipelineDesc.label = "forward_phong_pipeline";
    pipelineDesc.shader.name = "forward_phong_rhi";
    pipelineDesc.shader.stages = {
        {Backend::loadShaderSource(
             KE::getAssetPath("shaders/rhi/forward_material.vs")),
         Backend::ShaderType::Vertex, "main"},
        {Backend::loadShaderSource(
             KE::getAssetPath("shaders/rhi/forward_phong.fs")),
         Backend::ShaderType::Fragment, "main"},
    };
    pipelineDesc.pipelineLayout = _phongPipelineLayout.get();
    pipelineDesc.vertexBuffers = {position, transform, normal, texCoord, color,
                                  tangent};
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    _phongPipeline = _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "forward_phong_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _phongDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    // PBR Material Pipeline
    Backend::BindGroupLayoutDesc pbrMaterialLayoutDesc;
    pbrMaterialLayoutDesc.label = "pbr_material_group_layout";
    pbrMaterialLayoutDesc.entries.push_back(
        {0, Backend::BindingType::UniformBuffer,
         Backend::ShaderStageVisibility::Fragment});
    for (uint32_t binding = 1; binding <= 8; ++binding)
        pbrMaterialLayoutDesc.entries.push_back(
            {binding, Backend::BindingType::SampledTexture,
             Backend::ShaderStageVisibility::Fragment});
    pbrMaterialLayoutDesc.entries.push_back(
        {9, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment});
    _pbrMaterialGroupLayout =
        _graphicsDevice->createBindGroupLayout(pbrMaterialLayoutDesc);
    Backend::PipelineLayoutDesc pbrLayoutDesc;
    pbrLayoutDesc.label = "forward_pbr_pipeline_layout";
    pbrLayoutDesc.bindGroupLayouts = {
        _forwardGroupLayouts[0].get(), _shadowSamplingGroupLayout.get(),
        _forwardGroupLayouts[1].get(), _pbrMaterialGroupLayout.get()};
    _pbrPipelineLayout = _graphicsDevice->createPipelineLayout(pbrLayoutDesc);
    pipelineDesc.label = "forward_pbr_pipeline";
    pipelineDesc.shader.name = "forward_pbr_rhi";
    pipelineDesc.shader.stages[1] =
        {Backend::loadShaderSource(
             KE::getAssetPath("shaders/rhi/forward_pbr.fs")),
         Backend::ShaderType::Fragment, "main"};
    pipelineDesc.pipelineLayout = _pbrPipelineLayout.get();
    pipelineDesc.primitive.cullMode = Backend::CullMode::Back;
    _pbrPipeline = _graphicsDevice->createGraphicsPipeline(pipelineDesc);
    pipelineDesc.label = "forward_pbr_double_sided_pipeline";
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    _pbrDoubleSidedPipeline =
        _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    Backend::BlendState alphaBlend;
    alphaBlend.color.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    alphaBlend.color.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    alphaBlend.alpha.srcFactor = Backend::BlendFactorValue::SrcAlpha;
    alphaBlend.alpha.dstFactor = Backend::BlendFactorValue::OneMinusSrcAlpha;
    auto createTransparentVariants =
        [&](size_t kind, const char* label, Backend::PipelineLayout* layout,
            const std::string& vertexSource, const std::string& fragmentSource,
            std::vector<Backend::VertexBufferLayout> buffers) {
            Backend::GraphicsPipelineDesc desc;
            desc.label = std::string(label) + "_pipeline";
            desc.shader.name = std::string(label) + "_rhi";
            desc.shader.stages = {
                {vertexSource, Backend::ShaderType::Vertex, "main"},
                {fragmentSource, Backend::ShaderType::Fragment, "main"}};
            desc.pipelineLayout = layout;
            desc.vertexBuffers = std::move(buffers);
            desc.depthStencil = Backend::DepthStencilState{
                Backend::TextureFormat::Depth24Stencil8, false,
                Backend::CompareFunction::Less};
            desc.colorTargets = {{Backend::TextureFormat::RGBA16Float,
                                  alphaBlend}};
            desc.sampleCount = 4;
            desc.primitive.cullMode = Backend::CullMode::Back;
            _transparentPipelines[kind][0] =
                _graphicsDevice->createGraphicsPipeline(desc);
            desc.label = std::string(label) + "_double_sided_pipeline";
            desc.primitive.cullMode = Backend::CullMode::None;
            _transparentPipelines[kind][1] =
                _graphicsDevice->createGraphicsPipeline(desc);
        };
    const std::string vertexColorVs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_vertex_color.vs"));
    const std::string skinnedVertexColorVs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_skinned_vertex_color.vs"));
    const std::string materialVs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_material.vs"));
    const std::string vertexColorFs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_vertex_color.fs"));
    const std::string phongFs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_phong.fs"));
    const std::string pbrFs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_pbr.fs"));
    createTransparentVariants(0, "transparent_vertex_color",
                              _forwardPipelineLayout.get(), vertexColorVs,
                              vertexColorFs,
                              {position, transform, normal, empty, color});
    createTransparentVariants(
        1, "transparent_skinned_vertex_color", _forwardSkinPipelineLayout.get(),
        skinnedVertexColorVs, vertexColorFs,
        {position, transform, normal, empty, color, empty, boneIndices,
         boneWeights});
    createTransparentVariants(2, "transparent_phong",
                              _phongPipelineLayout.get(), materialVs, phongFs,
                              {position, transform, normal, texCoord, color,
                               tangent});
    createTransparentVariants(3, "transparent_pbr", _pbrPipelineLayout.get(),
                              materialVs, pbrFs,
                              {position, transform, normal, texCoord, color,
                               tangent});

    const std::string skinnedMaterialVs = Backend::loadShaderSource(
        KE::getAssetPath("shaders/rhi/forward_skinned_material.vs"));
    for (size_t model = 0; model < 2; ++model) {
        Backend::PipelineLayoutDesc skinnedLayoutDesc;
        skinnedLayoutDesc.label = model == 0
            ? "forward_skinned_phong_pipeline_layout"
            : "forward_skinned_pbr_pipeline_layout";
        skinnedLayoutDesc.bindGroupLayouts = {
            _forwardGroupLayouts[0].get(), _shadowSamplingGroupLayout.get(),
            _forwardSkinGroupLayout.get(),
            model == 0 ? _phongMaterialGroupLayout.get()
                       : _pbrMaterialGroupLayout.get()};
        _skinnedMaterialPipelineLayouts[model] =
            _graphicsDevice->createPipelineLayout(skinnedLayoutDesc);
        for (size_t transparent = 0; transparent < 2; ++transparent) {
            Backend::GraphicsPipelineDesc desc;
            const std::string baseLabel = std::string("forward_skinned_") +
                (model == 0 ? "phong" : "pbr") +
                (transparent ? "_transparent" : "_opaque");
            desc.label = baseLabel + "_pipeline";
            desc.shader.name = baseLabel + "_rhi";
            desc.shader.stages = {
                {skinnedMaterialVs, Backend::ShaderType::Vertex, "main"},
                {model == 0 ? phongFs : pbrFs,
                 Backend::ShaderType::Fragment, "main"}};
            desc.pipelineLayout = _skinnedMaterialPipelineLayouts[model].get();
            desc.vertexBuffers = {position, transform, normal, texCoord, color,
                                  tangent, boneIndices, boneWeights};
            desc.depthStencil = Backend::DepthStencilState{
                Backend::TextureFormat::Depth24Stencil8, !transparent,
                Backend::CompareFunction::Less};
            desc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
            if (transparent)
                desc.colorTargets[0].blend = alphaBlend;
            desc.sampleCount = 4;
            desc.primitive.cullMode = Backend::CullMode::Back;
            _skinnedMaterialPipelines[model][transparent][0] =
                _graphicsDevice->createGraphicsPipeline(desc);
            desc.label = baseLabel + "_double_sided_pipeline";
            desc.primitive.cullMode = Backend::CullMode::None;
            _skinnedMaterialPipelines[model][transparent][1] =
                _graphicsDevice->createGraphicsPipeline(desc);
        }
    }
}

bool Rasterizer::usesRhiTexturedVertexColor(
    const MeshInstancer& inst) const {
    return inst.material() &&
           inst.material()->shadingModel() ==
               MaterialShadingModel::VertexColor &&
           inst.shader() &&
           inst.shader()->getName().find("shaders/commonTex.fs") !=
               std::string::npos;
}

bool Rasterizer::usesRhiCheckerboard(const MeshInstancer& inst) const {
    return inst.material() &&
           inst.material()->shadingModel() ==
               MaterialShadingModel::VertexColor &&
           inst.shader() &&
           inst.shader()->getName().find("shaders/checkerboard.fs") !=
               std::string::npos;
}

void Rasterizer::setBackgroundSettings(const BackgroundSettings& settings) {
    if (!_checkerboardParamsBuffer)
        return;
    const std::array<glm::vec4, 4> params{
        settings.checkerColor1, settings.checkerColor2,
        settings.gridColor,
        glm::vec4(settings.showGrid ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f)};
    _checkerboardParamsBuffer->setData(params.data(), sizeof(params));
}

void Rasterizer::initSkyboxRhi() {
    static constexpr glm::vec3 vertices[] = {
        {-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1},
        {-1,-1, 1}, {1,-1, 1}, {1,1, 1}, {-1,1, 1}};
    static constexpr uint32_t indices[] = {
        0,1,2, 2,3,0, 1,5,6, 6,2,1, 5,4,7, 7,6,5,
        4,0,3, 3,7,4, 3,2,6, 6,7,3, 4,5,1, 1,0,4};
    _skyboxVertexBuffer = _graphicsDevice->createBuffer(
        Backend::BufferType::Vertex, sizeof(vertices), vertices);
    _skyboxIndexBuffer = _graphicsDevice->createBuffer(
        Backend::BufferType::Index, sizeof(indices), indices);
    Backend::BufferDesc paramsDesc;
    paramsDesc.size = sizeof(glm::vec4);
    paramsDesc.usage = Backend::BufferUsage::Uniform |
                       Backend::BufferUsage::CopyDst;
    paramsDesc.label = "skybox_params";
    _skyboxParamsBuffer = _graphicsDevice->createBuffer(paramsDesc);

    Backend::BindGroupLayoutDesc passLayoutDesc;
    passLayoutDesc.label = "skybox_pass_group_layout";
    passLayoutDesc.entries = {{0, Backend::BindingType::UniformBuffer,
                               Backend::ShaderStageVisibility::Vertex}};
    _skyboxPassGroupLayout =
        _graphicsDevice->createBindGroupLayout(passLayoutDesc);
    Backend::BindGroupDesc paramsGroupDesc;
    paramsGroupDesc.layout = _skyboxPassGroupLayout.get();
    paramsGroupDesc.label = "skybox_params_bind_group";
    paramsGroupDesc.entries = {{0, _skyboxParamsBuffer.get(), 0,
                                sizeof(glm::vec4), nullptr, nullptr}};
    _skyboxParamsBindGroup =
        _graphicsDevice->createBindGroup(paramsGroupDesc);

    Backend::BindGroupLayoutDesc textureLayoutDesc;
    textureLayoutDesc.label = "skybox_texture_group_layout";
    textureLayoutDesc.entries = {
        {0, Backend::BindingType::SampledTexture,
         Backend::ShaderStageVisibility::Fragment,
         Backend::TextureFormat::Undefined,
         Backend::TextureSampleType::Float,
         Backend::TextureViewDimension::Cube},
        {1, Backend::BindingType::Sampler,
         Backend::ShaderStageVisibility::Fragment}};
    _skyboxTextureGroupLayout =
        _graphicsDevice->createBindGroupLayout(textureLayoutDesc);
    Backend::PipelineLayoutDesc pipelineLayoutDesc;
    pipelineLayoutDesc.label = "skybox_pipeline_layout";
    pipelineLayoutDesc.bindGroupLayouts = {
        _forwardGroupLayouts[0].get(), _skyboxPassGroupLayout.get(),
        _forwardGroupLayouts[1].get(), _skyboxTextureGroupLayout.get()};
    _skyboxPipelineLayout =
        _graphicsDevice->createPipelineLayout(pipelineLayoutDesc);

    Backend::VertexBufferLayout vertexLayout;
    vertexLayout.arrayStride = sizeof(glm::vec3);
    vertexLayout.attributes = {{Backend::VertexFormat::Float32x3, 0, 0}};
    Backend::GraphicsPipelineDesc pipelineDesc;
    pipelineDesc.label = "skybox_pipeline";
    pipelineDesc.shader.name = "skybox_rhi";
    pipelineDesc.shader.stages = {
        {Backend::loadShaderSource(KE::getAssetPath("shaders/rhi/skybox.vs")),
         Backend::ShaderType::Vertex, "main"},
        {Backend::loadShaderSource(KE::getAssetPath("shaders/rhi/skybox.fs")),
         Backend::ShaderType::Fragment, "main"}};
    pipelineDesc.pipelineLayout = _skyboxPipelineLayout.get();
    pipelineDesc.vertexBuffers = {vertexLayout};
    pipelineDesc.primitive.cullMode = Backend::CullMode::None;
    pipelineDesc.depthStencil = Backend::DepthStencilState{
        Backend::TextureFormat::Depth24Stencil8, false,
        Backend::CompareFunction::LessEqual};
    pipelineDesc.colorTargets = {{Backend::TextureFormat::RGBA16Float}};
    pipelineDesc.sampleCount = 4;
    _skyboxPipeline = _graphicsDevice->createGraphicsPipeline(pipelineDesc);

    Backend::SamplerDesc samplerDesc;
    samplerDesc.wrapU = Backend::TextureWrap::ClampToEdge;
    samplerDesc.wrapV = Backend::TextureWrap::ClampToEdge;
    samplerDesc.minFilter = Backend::TextureFilter::Linear;
    samplerDesc.magFilter = Backend::TextureFilter::Linear;
    samplerDesc.label = "skybox_sampler";
    _skyboxSampler = _graphicsDevice->createSampler(samplerDesc);
}

void Rasterizer::rebuildSkyboxBinding(UpAxis upAxis) {
    const glm::vec4 params{upAxis == UpAxis::Z ? 1.0f : 0.0f, 0.0f, 0.0f,
                           0.0f};
    _skyboxParamsBuffer->setData(&params, sizeof(params));
    _skyboxTextureBindGroup.reset();
    _skyboxTextureView.reset();
    if (!_skyboxTexture)
        return;
    Backend::TextureViewDesc viewDesc;
    viewDesc.dimension = Backend::TextureViewDimension::Cube;
    viewDesc.arrayLayerCount = 6;
    viewDesc.label = "skybox_cube_view";
    _skyboxTextureView =
        _graphicsDevice->createTextureView(_skyboxTexture.get(), viewDesc);
    Backend::BindGroupDesc groupDesc;
    groupDesc.layout = _skyboxTextureGroupLayout.get();
    groupDesc.label = "skybox_texture_bind_group";
    groupDesc.entries = {
        {0, nullptr, 0, 0, _skyboxTextureView.get(), nullptr},
        {1, nullptr, 0, 0, nullptr, _skyboxSampler.get()}};
    _skyboxTextureBindGroup = _graphicsDevice->createBindGroup(groupDesc);
}

void Rasterizer::setSkybox(const std::string& path, UpAxis upAxis) {
    _skyboxTexture = _graphicsDevice->createCubemapTexture(path);
    rebuildSkyboxBinding(upAxis);
}

void Rasterizer::setSkybox(const std::vector<std::string>& paths,
                           UpAxis upAxis) {
    _skyboxTexture = _graphicsDevice->createCubemapTexture(paths);
    rebuildSkyboxBinding(upAxis);
}

Backend::BindGroup*
Rasterizer::updatePhongRhiResources(PhongMaterial& material,
                                    const MeshInstancer& inst) {
    struct alignas(16) PhongParams {
        glm::vec4 ambientShininess;
        glm::vec4 diffuseAlphaCutoff;
        glm::vec4 specularAlphaMode;
        glm::vec4 textureFlags;
    };
    auto [it, inserted] = _phongRhiResources.try_emplace(&material);
    PhongRhiResources& resources = it->second;
    if (inserted) {
        Backend::BufferDesc desc;
        desc.size = sizeof(PhongParams);
        desc.usage = Backend::BufferUsage::Uniform |
                     Backend::BufferUsage::CopyDst;
        desc.label = "phong_material_params";
        resources.params = _graphicsDevice->createBuffer(desc);
    }
    const PhongParams params{
        glm::vec4(material.ambient, material.shininess),
        glm::vec4(material.diffuse, inst.alphaCutoff()),
        glm::vec4(material.specular,
                  static_cast<float>(static_cast<int>(inst.alphaMode()))),
        glm::vec4(material.diffuseMap ? 1.0f : 0.0f,
                  material.specularMap ? 1.0f : 0.0f,
                  material.alphaMap ? 1.0f : 0.0f,
                  material.normalMap ? 1.0f : 0.0f)};
    resources.params->setData(&params, sizeof(params));

    std::array<Backend::Texture*, 4> textures{
        material.diffuseMap ? material.diffuseMap : _materialWhiteTexture.get(),
        material.specularMap ? material.specularMap : _materialWhiteTexture.get(),
        material.alphaMap ? material.alphaMap : _materialWhiteTexture.get(),
        material.normalMap ? material.normalMap : _materialNormalTexture.get()};
    std::array<uintptr_t, 4> handles{};
    for (size_t i = 0; i < textures.size(); ++i)
        handles[i] = textures[i]->getNativeHandle();
    if (!resources.bindGroup || handles != resources.textureHandles) {
        resources.bindGroup.reset();
        for (size_t i = 0; i < textures.size(); ++i) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.label = "phong_material_view_" + std::to_string(i);
            resources.views[i] =
                _graphicsDevice->createTextureView(textures[i], viewDesc);
        }
        Backend::BindGroupDesc desc;
        desc.layout = _phongMaterialGroupLayout.get();
        desc.label = "phong_material_bind_group";
        desc.entries = {
            {0, resources.params.get(), 0, sizeof(PhongParams), nullptr, nullptr},
            {1, nullptr, 0, 0, resources.views[0].get(), nullptr},
            {2, nullptr, 0, 0, resources.views[1].get(), nullptr},
            {3, nullptr, 0, 0, resources.views[2].get(), nullptr},
            {4, nullptr, 0, 0, resources.views[3].get(), nullptr},
            {5, nullptr, 0, 0, nullptr, _materialSampler.get()},
        };
        resources.bindGroup = _graphicsDevice->createBindGroup(desc);
        resources.textureHandles = handles;
    }
    return resources.bindGroup.get();
}

Backend::BindGroup*
Rasterizer::updatePbrRhiResources(PBRMaterial& material,
                                  const MeshInstancer& inst) {
    struct alignas(16) PbrParams {
        glm::vec4 baseColor;
        glm::vec4 factors;
        glm::vec4 emissiveAlpha;
        glm::vec4 textureFlags0;
        glm::vec4 textureFlags1;
    };
    auto [it, inserted] = _pbrRhiResources.try_emplace(&material);
    PbrRhiResources& resources = it->second;
    if (inserted) {
        Backend::BufferDesc desc;
        desc.size = sizeof(PbrParams);
        desc.usage = Backend::BufferUsage::Uniform |
                     Backend::BufferUsage::CopyDst;
        desc.label = "pbr_material_params";
        resources.params = _graphicsDevice->createBuffer(desc);
    }
    const PbrParams params{
        material.baseColor,
        glm::vec4(material.metallic, material.roughness,
                  material.emissiveStrength, inst.alphaCutoff()),
        glm::vec4(material.emissiveColor,
                  static_cast<float>(static_cast<int>(inst.alphaMode()))),
        glm::vec4(material.baseColorTexture ? 1.0f : 0.0f,
                  material.normalTexture ? 1.0f : 0.0f,
                  material.metallicRoughnessTexture ? 1.0f : 0.0f,
                  material.metallicTexture ? 1.0f : 0.0f),
        glm::vec4(material.roughnessTexture ? 1.0f : 0.0f,
                  material.aoTexture ? 1.0f : 0.0f,
                  material.ormTexture ? 1.0f : 0.0f,
                  material.emissiveTexture ? 1.0f : 0.0f)};
    resources.params->setData(&params, sizeof(params));

    std::array<Backend::Texture*, 8> textures{
        material.baseColorTexture ? material.baseColorTexture
                                  : _materialWhiteTexture.get(),
        material.normalTexture ? material.normalTexture
                               : _materialNormalTexture.get(),
        material.metallicRoughnessTexture
            ? material.metallicRoughnessTexture
            : _materialWhiteTexture.get(),
        material.metallicTexture ? material.metallicTexture
                                 : _materialWhiteTexture.get(),
        material.roughnessTexture ? material.roughnessTexture
                                  : _materialWhiteTexture.get(),
        material.aoTexture ? material.aoTexture : _materialWhiteTexture.get(),
        material.ormTexture ? material.ormTexture : _materialWhiteTexture.get(),
        material.emissiveTexture ? material.emissiveTexture
                                 : _materialWhiteTexture.get()};
    std::array<uintptr_t, 8> handles{};
    for (size_t i = 0; i < textures.size(); ++i)
        handles[i] = textures[i]->getNativeHandle();
    if (!resources.bindGroup || handles != resources.textureHandles) {
        resources.bindGroup.reset();
        for (size_t i = 0; i < textures.size(); ++i) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.label = "pbr_material_view_" + std::to_string(i);
            resources.views[i] =
                _graphicsDevice->createTextureView(textures[i], viewDesc);
        }
        Backend::BindGroupDesc desc;
        desc.layout = _pbrMaterialGroupLayout.get();
        desc.label = "pbr_material_bind_group";
        desc.entries.push_back(
            {0, resources.params.get(), 0, sizeof(PbrParams), nullptr, nullptr});
        for (uint32_t binding = 1; binding <= 8; ++binding)
            desc.entries.push_back(
                {binding, nullptr, 0, 0,
                 resources.views[binding - 1].get(), nullptr});
        desc.entries.push_back(
            {9, nullptr, 0, 0, nullptr, _materialSampler.get()});
        resources.bindGroup = _graphicsDevice->createBindGroup(desc);
        resources.textureHandles = handles;
    }
    return resources.bindGroup.get();
}

Backend::BindGroup* Rasterizer::updateTexturedVertexColorRhiResources(
    const MeshInstancer& inst) {
    auto [it, inserted] =
        _texturedVertexColorRhiResources.try_emplace(&inst);
    TexturedVertexColorRhiResources& resources = it->second;
    if (inserted) {
        Backend::BufferDesc desc;
        desc.size = sizeof(glm::vec4);
        desc.usage = Backend::BufferUsage::Uniform |
                     Backend::BufferUsage::CopyDst;
        desc.label = "textured_vertex_color_params";
        resources.params = _graphicsDevice->createBuffer(desc);
    }

    Backend::Texture* baseColor =
        inst.textureAtSlot(RendererTextureSlot::BaseColor);
    Backend::Texture* normal =
        inst.textureAtSlot(RendererTextureSlot::Normal);
    const glm::vec4 params{
        normal && inst.hasTangents() ? 1.0f : 0.0f,
        static_cast<float>(static_cast<int>(inst.alphaMode())),
        inst.alphaCutoff(), 0.0f};
    resources.params->setData(&params, sizeof(params));

    std::array<Backend::Texture*, 2> textures{
        baseColor ? baseColor : _materialWhiteTexture.get(),
        normal ? normal : _materialNormalTexture.get()};
    const std::array<const Backend::Texture*, 2> textureIdentity{
        textures[0], textures[1]};
    if (!resources.bindGroup || resources.textures != textureIdentity) {
        resources.bindGroup.reset();
        for (size_t i = 0; i < textures.size(); ++i) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.label =
                "textured_vertex_color_view_" + std::to_string(i);
            resources.views[i] =
                _graphicsDevice->createTextureView(textures[i], viewDesc);
        }
        Backend::BindGroupDesc desc;
        desc.layout = _texturedVertexColorGroupLayout.get();
        desc.label = "textured_vertex_color_bind_group";
        desc.entries = {
            {0, nullptr, 0, 0, resources.views[0].get(), nullptr},
            {1, nullptr, 0, 0, resources.views[1].get(), nullptr},
            {2, nullptr, 0, 0, nullptr, _materialSampler.get()},
            {3, resources.params.get(), 0, sizeof(glm::vec4), nullptr,
             nullptr},
        };
        resources.bindGroup = _graphicsDevice->createBindGroup(desc);
        resources.textures = textureIdentity;
    }
    return resources.bindGroup.get();
}

// -------------------------------------------------------------------------
// Selection Mask Pipeline - target lifetime and command submission
// -------------------------------------------------------------------------
void Rasterizer::ensureSelectionMaskTarget(Backend::Framebuffer* target) {
    Backend::Texture* texture = target ? target->getColorTexture() : nullptr;
    if (!texture)
        return;
    const uintptr_t handle = texture->getNativeHandle();
    if (_selectionMaskOutputTarget && _selectionMaskOutputHandle == handle &&
        _selectionMaskOutputWidth == texture->getWidth() &&
        _selectionMaskOutputHeight == texture->getHeight())
        return;
    _selectionMaskOutputTarget.reset();
    _selectionMaskOutputView.reset();
    Backend::TextureViewDesc viewDesc;
    viewDesc.format = Backend::TextureFormat::RGBA8Unorm;
    viewDesc.label = "selection_mask_output_view";
    _selectionMaskOutputView =
        _graphicsDevice->createTextureView(texture, viewDesc);
    Backend::RenderPassDesc passDesc;
    passDesc.label = "selection_mask_opaque_pass";
    passDesc.colorAttachments = {{_selectionMaskOutputView.get(), nullptr,
                                  Backend::LoadOp::Clear,
                                  Backend::StoreOp::Store,
                                  {0.0f, 0.0f, 0.0f, 1.0f}}};
    _selectionMaskOutputTarget =
        _graphicsDevice->createRenderTarget(passDesc);
    _selectionMaskOutputHandle = handle;
    _selectionMaskOutputWidth = texture->getWidth();
    _selectionMaskOutputHeight = texture->getHeight();
}

void Rasterizer::renderSelectionMaskPass(const RayPickResult& selection,
                                         Backend::Framebuffer* target,
                                         int width, int height) {
    if (!target || !selection.hit || selection.handle >= _handleTable.size())
        return;

    MeshInstancer* inst = _handleTable[selection.handle];
    if (!inst || selection.instanceIndex < 0)
        return;

    {
        ensureSelectionMaskTarget(target);
        if (!_selectionMaskOutputTarget || width <= 0 || height <= 0)
            return;
        Backend::Texture* alphaTexture =
            inst->alphaMode() == AlphaMode::Mask ? inst->alphaMaskTexture()
                                                 : nullptr;
        std::unique_ptr<Backend::TextureView> alphaView;
        std::unique_ptr<Backend::BindGroup> alphaBindGroup;
        std::unique_ptr<Backend::BindGroup> skinBindGroup;
        if (alphaTexture) {
            Backend::TextureViewDesc alphaViewDesc;
            alphaViewDesc.format = alphaTexture->getFormat();
            alphaViewDesc.label = "selection_mask_alpha_view";
            alphaView = _graphicsDevice->createTextureView(alphaTexture,
                                                           alphaViewDesc);
            if (!inst->alphaParamsBuffer())
                throw std::runtime_error(
                    "RHI alpha-mask draw is missing its parameter buffer");
            Backend::BindGroupDesc alphaBindDesc;
            alphaBindDesc.layout = _selectionMaskAlphaGroupLayout.get();
            alphaBindDesc.label = "selection_mask_alpha_bind_group";
            alphaBindDesc.entries = {
                {0, nullptr, 0, 0, alphaView.get(), nullptr},
                {1, nullptr, 0, 0, nullptr,
                 _selectionMaskAlphaSampler.get()},
                {2, inst->alphaParamsBuffer(), 0,
                 sizeof(glm::vec4), nullptr, nullptr},
            };
            alphaBindGroup =
                _graphicsDevice->createBindGroup(alphaBindDesc);
        }
        if (inst->hasSkinning()) {
            if (!inst->boneMatricesBuffer())
                throw std::runtime_error(
                    "RHI skinned selection mask is missing its bone buffer");
            Backend::BindGroupDesc skinBindDesc;
            skinBindDesc.layout = _selectionMaskSkinGroupLayout.get();
            skinBindDesc.label = "selection_mask_skin_bind_group";
            skinBindDesc.entries = {{
                0, inst->boneMatricesBuffer(), 0,
                sizeof(glm::mat4) * Scene::MaxSkinningBones, nullptr, nullptr}};
            skinBindGroup = _graphicsDevice->createBindGroup(skinBindDesc);
        }
        auto encoder = _graphicsDevice->createCommandEncoder();
        auto pass =
            encoder->beginRenderPass(_selectionMaskOutputTarget.get());
        pass->setViewport(0.0f, 0.0f, static_cast<float>(width),
                          static_cast<float>(height));
        if (inst->hasSkinning() && alphaTexture)
            pass->setPipeline(
                inst->isDoubleSided()
                    ? _selectionMaskSkinAlphaDoubleSidedPipeline.get()
                    : _selectionMaskSkinAlphaPipeline.get());
        else if (inst->hasSkinning())
            pass->setPipeline(
                inst->isDoubleSided()
                    ? _selectionMaskSkinDoubleSidedPipeline.get()
                    : _selectionMaskSkinPipeline.get());
        else if (alphaTexture)
            pass->setPipeline(
                inst->isDoubleSided()
                    ? _selectionMaskAlphaDoubleSidedPipeline.get()
                    : _selectionMaskAlphaPipeline.get());
        else
            pass->setPipeline(inst->isDoubleSided()
                                  ? _selectionMaskDoubleSidedPipeline.get()
                                  : _selectionMaskPipeline.get());
        pass->setBindGroup(0, _selectionMaskFrameBindGroup.get());
        if (skinBindGroup)
            pass->setBindGroup(2, skinBindGroup.get());
        if (alphaBindGroup)
            pass->setBindGroup(3, alphaBindGroup.get());
        inst->recordInstanceMask(*pass, selection.instanceIndex,
                                 alphaTexture != nullptr,
                                 inst->hasSkinning());
        pass->end();
        auto commands = encoder->finish();
        _graphicsDevice->submit(*commands);
        return;
    }
}

// -------------------------------------------------------------------------
// Shadow Depth Pipeline - frame data, CSM setup, and command submission
// -------------------------------------------------------------------------

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
    if (!_shadowRenderTarget || _shadowDistance <= 0.0f) {
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

            const int mapSize = _cascadeMapSizes[static_cast<size_t>(i)];
            drawShadowCasters(
                _cascadeRenderTargets[static_cast<size_t>(i)].get(), mapSize);
        }
        _lightSpaceMatrix = _cascadeLightMatrices[0];
        _shadowMap = _cascadeMaps[0];
        updateShadowUBO(_shadowMap ? _activeShadowOrthoHalfSize : 0.0f);
    } else {
        updateShadowPassUBO(_lightSpaceMatrix, _activeShadowOrthoHalfSize);
        drawShadowCasters(_shadowRenderTarget.get(), _shadowMapWH);
    }
    _graphicsDevice->setViewport(0, 0, viewportWidth, viewportHeight);
}

void Rasterizer::drawShadowCasters(Backend::RenderTarget* target,
                                   int mapSize) {
    if (!target || mapSize <= 0)
        return;
    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(target);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(mapSize),
                      static_cast<float>(mapSize));

    // Views and bind groups must outlive command submission. Keeping them per
    // pass avoids shared mutable material bindings during future parallel
    // command recording.
    std::vector<std::unique_ptr<Backend::TextureView>> alphaViews;
    std::vector<std::unique_ptr<Backend::BindGroup>> alphaBindGroups;
    std::vector<std::unique_ptr<Backend::BindGroup>> skinBindGroups;
    for (auto& [key, inst] : _instancers) {
        if (inst.visibleCount() == 0)
            continue;
        if (!inst.castsShadow())
            continue;
        Backend::Texture* alphaTexture =
            inst.alphaMode() == AlphaMode::Mask ? inst.alphaMaskTexture()
                                                : nullptr;
        Backend::BindGroup* alphaBindGroup = nullptr;
        Backend::BindGroup* skinBindGroup = nullptr;
        if (alphaTexture) {
            Backend::TextureViewDesc viewDesc;
            viewDesc.format = alphaTexture->getFormat();
            viewDesc.label = "shadow_alpha_view";
            alphaViews.push_back(
                _graphicsDevice->createTextureView(alphaTexture, viewDesc));
            Backend::BindGroupDesc bindDesc;
            bindDesc.layout = _selectionMaskAlphaGroupLayout.get();
            bindDesc.label = "shadow_alpha_bind_group";
            bindDesc.entries = {
                {0, nullptr, 0, 0, alphaViews.back().get(), nullptr},
                {1, nullptr, 0, 0, nullptr,
                 _selectionMaskAlphaSampler.get()},
                {2, inst.alphaParamsBuffer(), 0, sizeof(glm::vec4), nullptr,
                 nullptr},
            };
            alphaBindGroups.push_back(
                _graphicsDevice->createBindGroup(bindDesc));
            alphaBindGroup = alphaBindGroups.back().get();
        }
        if (inst.hasSkinning()) {
            Backend::BindGroupDesc bindDesc;
            bindDesc.layout = _selectionMaskSkinGroupLayout.get();
            bindDesc.label = "shadow_skin_bind_group";
            bindDesc.entries = {{
                0, inst.boneMatricesBuffer(), 0,
                sizeof(glm::mat4) * Scene::MaxSkinningBones, nullptr, nullptr}};
            skinBindGroups.push_back(
                _graphicsDevice->createBindGroup(bindDesc));
            skinBindGroup = skinBindGroups.back().get();
        }
        pass->setPipeline(
            _shadowPipelines[shadowPipelineIndex(
                                 inst.hasSkinning(), alphaTexture != nullptr,
                                 inst.isDoubleSided())]
                .get());
        pass->setBindGroup(0, _shadowFrameBindGroup.get());
        if (skinBindGroup)
            pass->setBindGroup(2, skinBindGroup);
        if (alphaBindGroup)
            pass->setBindGroup(3, alphaBindGroup);
        inst.recordDraw(*pass, alphaTexture != nullptr, inst.hasSkinning());
    }
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

} // namespace KE

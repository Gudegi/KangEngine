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
    _forwardPass.initialize(_graphicsDevice);
    _shadowPass.initialize(_graphicsDevice);
    _skyboxPass.initialize(_graphicsDevice);
    _selectionMaskPass.initialize(_graphicsDevice);
    Backend::BufferDesc uniformDesc;
    uniformDesc.size = 2 * sizeof(glm::mat4);
    uniformDesc.usage =
        Backend::BufferUsage::Uniform | Backend::BufferUsage::CopyDst;
    uniformDesc.label = "camera_uniforms";
    _cameraUBO = _graphicsDevice->createBuffer(uniformDesc);
    _graphicsDevice->bindUniformBuffer(_cameraUBO.get(), CAMERA_UBO_BIND_SLOT);
    uniformDesc.size = LIGHT_UBO_VEC4_COUNT * sizeof(glm::vec4);
    uniformDesc.label = "light_uniforms";
    _lightUBO = _graphicsDevice->createBuffer(uniformDesc);
    _graphicsDevice->bindUniformBuffer(_lightUBO.get(), LIGHT_UBO_BIND_SLOT);
    // std140 shadow data: mat4[4] cascade matrices, vec4 cascade splits,
    // vec4 cascade ortho half-sizes, vec4 cascade map sizes, vec4 params,
    // vec4 info.
    uniformDesc.size =
        MaxShadowCascades * sizeof(glm::mat4) + 5 * sizeof(glm::vec4);
    uniformDesc.label = "shadow_uniforms";
    _shadowUBO = _graphicsDevice->createBuffer(uniformDesc);
    _graphicsDevice->bindUniformBuffer(_shadowUBO.get(), SHADOW_UBO_BIND_SLOT);

    _shadowFbo = _graphicsDevice->createFramebuffer(
        {_shadowMapWH, _shadowMapWH, true, false, 0}); // depth-only, no MSAA
    for (size_t i = 0; i < _cascadeFbos.size(); ++i) {
        const int mapSize = _cascadeMapSizes[i];
        auto& fbo = _cascadeFbos[i];
        fbo = _graphicsDevice->createFramebuffer(
            {mapSize, mapSize, true, false, 0});
    }
    _selectionMaskPass.initializeResources(_cameraUBO.get(), _shaderLibrary);
    initShadowRhi();
    initForwardRhi();
    _skyboxPass.initializeResources(_forwardPass.frameLayout(), _shaderLibrary);
    _debugRenderer.init(_graphicsDevice, _forwardPass.frameLayout(),
                        _forwardPass.frameBindGroup(), &_shaderLibrary);
    _textRenderer.init(
        _graphicsDevice,
        FontAtlasData::loadAscii(KE::getAssetPath("fonts/godoFont/GodoM.ttf")),
        _forwardPass.frameLayout(), _forwardPass.frameBindGroup(),
        &_shaderLibrary);
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

    InstancerKey key{meshData.get(), material, transformSource};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, skinnedMesh, transformSource,
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

RenderableHandle Rasterizer::addRenderable(Material* material,
                                           Scene::Prim* prim,
                                           TransformSource transformSource) {
    auto meshData = prim->resolveMeshData();
    if (!material || !meshData || meshData->vertices.empty() ||
        meshData->indices.empty())
        return InvalidHandle;

    InstancerKey key{meshData.get(), material, transformSource};
    auto it = _instancers.find(key);
    if (it == _instancers.end()) {
        auto [newIt, inserted] = _instancers.emplace(key, MeshInstancer{});
        newIt->second.init(_graphicsDevice, *meshData, transformSource,
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
                                float size, bool hidden, bool overlay) {
    _debugRenderer.logPoints(path, points, colors, size, hidden, overlay);
}

void Rasterizer::clearDebugPoints(const std::string& path) {
    _debugRenderer.clearPoints(path);
}

void Rasterizer::setWorldText(const std::string& path,
                              const WorldTextDesc& desc) {
    _textRenderer.setWorldText(path, desc);
}

void Rasterizer::setWorldTextString(const std::string& path, std::string text) {
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
        const auto found = std::find_if(hooks.begin(), hooks.end(),
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

void Rasterizer::recordRenderHooks(RenderHookPhase phase,
                                   Backend::RenderTarget* sceneDrawTarget) {
    if (!sceneDrawTarget)
        return;
    const size_t phaseIndex = static_cast<size_t>(phase);
    if (phaseIndex >= _renderHooks.size() || _renderHooks[phaseIndex].empty())
        return;

    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(sceneDrawTarget);
    RenderHookContext context{*pass, *sceneDrawTarget,
                              _forwardPass.frameBindGroup(), _viewportWidth,
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
    if (!sceneDrawTarget || !_forwardPass.frameBindGroup() ||
        !_shadowPass.samplingBindGroup())
        return;

    std::vector<MeshInstancer*> drawables;
    drawables.reserve(_instancers.size());
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
        drawables.push_back(&inst);
    }
    auto prepared = _forwardPass.prepare(drawables, false);

    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(sceneDrawTarget);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(_viewportWidth),
                      static_cast<float>(_viewportHeight));
    pass->setPolygonMode(_wireframeEnabled ? Backend::PolygonMode::Line
                                           : Backend::PolygonMode::Fill);
    _forwardPass.record(*pass, prepared, _shadowPass.samplingBindGroup());
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

// -------------------------------------------------------------------------
// Skybox Pipeline
// -------------------------------------------------------------------------
void Rasterizer::renderSkyboxPass(const glm::mat4& view, const glm::mat4& proj,
                                  Backend::RenderTarget* sceneDrawTarget) {
    // Drawn after opaque geometry so the skybox only fills empty pixels.
    if (!sceneDrawTarget || !_skyboxPass.ready()) {
        return;
    }
    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(sceneDrawTarget);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(_viewportWidth),
                      static_cast<float>(_viewportHeight));
    _skyboxPass.record(*pass, _forwardPass.frameBindGroup());
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

// -------------------------------------------------------------------------
// Transparent Scene Pipeline
// -------------------------------------------------------------------------
void Rasterizer::renderTransparentPass(Backend::Texture* shadowTexture,
                                       Backend::RenderTarget* sceneDrawTarget) {
    if (!sceneDrawTarget || !_forwardPass.frameBindGroup() ||
        !_shadowPass.samplingBindGroup())
        return;

    std::vector<MeshInstancer*> drawables;
    drawables.reserve(_instancers.size());
    for (auto& entry : _instancers) {
        MeshInstancer& inst = entry.second;
        if (inst.hasTransparent() && inst.visibleCount() > 0)
            drawables.push_back(&inst);
    }
    auto prepared = _forwardPass.prepare(drawables, true);

    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(sceneDrawTarget);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(_viewportWidth),
                      static_cast<float>(_viewportHeight));
    pass->setPolygonMode(_wireframeEnabled ? Backend::PolygonMode::Line
                                           : Backend::PolygonMode::Fill);
    _forwardPass.record(*pass, prepared, _shadowPass.samplingBindGroup());
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

// -------------------------------------------------------------------------
// Debug Overlay Pipeline
// -------------------------------------------------------------------------
void Rasterizer::renderDebugOverlayPass(
    Backend::RenderTarget* sceneDrawTarget) {
    updateDebugRenderAABB();
    _textRenderer.prepare(_viewportWidth, _viewportHeight);
    if (!sceneDrawTarget ||
        (!_debugRenderer.hasDraws() && !_textRenderer.hasDraws()))
        return;
    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(sceneDrawTarget);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(_viewportWidth),
                      static_cast<float>(_viewportHeight));
    _debugRenderer.record(*pass);
    _textRenderer.record(*pass);
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

void Rasterizer::initShadowRhi() {
    // =====================================================================
    // Shadow Depth Pipeline
    // Records single-map and CSM depth passes. The 8 immutable variants cover
    // static/skinned, opaque/alpha-mask, and culled/double-sided casters.
    // =====================================================================
    _shadowPass.initializeResources(_shadowUBO.get(), _shaderLibrary);

    // ShadowPass owns the RHI views and render targets. Framebuffers remain
    // here temporarily as the OpenGL shadow-texture allocation boundary.
    std::array<Backend::Framebuffer*, MaxShadowCascades> cascadeFbos{};
    for (size_t i = 0; i < cascadeFbos.size(); ++i)
        cascadeFbos[i] = _cascadeFbos[i].get();
    _shadowPass.initializeDepthTargets(_shadowFbo.get(), cascadeFbos);

    // Forward and deferred lighting consume shadow maps through this group.
    _shadowPass.initializeSampling();
    rebuildShadowSamplingBindings(activeShadowTexture());
}

void Rasterizer::rebuildShadowSamplingBindings(
    Backend::Texture* fallbackTexture) {
    if (!fallbackTexture)
        return;
    const bool hasShadow = _shadowMap && _shadowDistance > 0.0f;
    std::array<Backend::Texture*, MaxShadowCascades> textures{};
    for (size_t i = 0; i < textures.size(); ++i)
        textures[i] = (_useCsm && hasShadow && _cascadeMaps[i])
                          ? _cascadeMaps[i]
                          : fallbackTexture;
    _shadowPass.rebuildSampling(textures, _debugCsmCascadeTint && _useCsm);
}

void Rasterizer::initForwardRhi() {
    _forwardPass.initializeResources(
        _cameraUBO.get(), _lightUBO.get(), _shadowUBO.get(),
        _shadowPass.samplingLayout(), _shaderLibrary);
}
void Rasterizer::setBackgroundSettings(const BackgroundSettings& settings) {
    _forwardPass.setBackgroundSettings(settings);
}

void Rasterizer::setSkybox(const std::string& path, UpAxis upAxis) {
    _skyboxPass.setTexture(path, upAxis);
}

void Rasterizer::setSkybox(const std::vector<std::string>& paths,
                           UpAxis upAxis) {
    _skyboxPass.setTexture(paths, upAxis);
}

// -------------------------------------------------------------------------
// Selection Mask Pipeline - target lifetime and command submission
// -------------------------------------------------------------------------
void Rasterizer::renderSelectionMaskPass(const RayPickResult& selection,
                                         Backend::Framebuffer* target,
                                         int width, int height) {
    if (!target || !selection.hit || selection.handle >= _handleTable.size())
        return;

    MeshInstancer* inst = _handleTable[selection.handle];
    if (!inst || selection.instanceIndex < 0)
        return;

    if (width <= 0 || height <= 0)
        return;
    auto prepared =
        _selectionMaskPass.prepare(inst, selection.instanceIndex, target);
    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(_selectionMaskPass.target());
    pass->setViewport(0.0f, 0.0f, static_cast<float>(width),
                      static_cast<float>(height));
    _selectionMaskPass.record(*pass, prepared);
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
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
    if (!_shadowPass.singleDepthTarget() || _shadowDistance <= 0.0f) {
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
                _shadowPass.cascadeDepthTarget(static_cast<size_t>(i)),
                mapSize);
        }
        _lightSpaceMatrix = _cascadeLightMatrices[0];
        _shadowMap = _cascadeMaps[0];
        updateShadowUBO(_shadowMap ? _activeShadowOrthoHalfSize : 0.0f);
    } else {
        updateShadowPassUBO(_lightSpaceMatrix, _activeShadowOrthoHalfSize);
        drawShadowCasters(_shadowPass.singleDepthTarget(), _shadowMapWH);
    }
    _graphicsDevice->setViewport(0, 0, viewportWidth, viewportHeight);
}

void Rasterizer::drawShadowCasters(Backend::RenderTarget* target, int mapSize) {
    if (!target || mapSize <= 0)
        return;

    std::vector<MeshInstancer*> casters;
    casters.reserve(_instancers.size());
    for (auto& [key, inst] : _instancers) {
        if (inst.visibleCount() > 0 && inst.castsShadow())
            casters.push_back(&inst);
    }
    auto prepared = _shadowPass.prepare(casters);

    auto encoder = _graphicsDevice->createCommandEncoder();
    auto pass = encoder->beginRenderPass(target);
    pass->setViewport(0.0f, 0.0f, static_cast<float>(mapSize),
                      static_cast<float>(mapSize));
    _shadowPass.record(*pass, prepared);
    pass->end();
    auto commands = encoder->finish();
    _graphicsDevice->submit(*commands);
}

} // namespace KE

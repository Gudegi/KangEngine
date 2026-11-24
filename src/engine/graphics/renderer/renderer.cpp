#include "renderer.hpp"
#include "engine/graphics/renderer/render_backend/native/native_render_system.hpp"

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/scene/types/camera.hpp"
#include "engine/graphics/renderer/render_backend/native/native_render_extension.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/scene_backend.hpp"

#include <stdexcept>
#include <algorithm>
#include <utility>

namespace KE {

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

void Renderer::initialize(Backend::GraphicsDevice& device, int width,
                          int height, std::unique_ptr<RenderBackend> backend) {
    if (!backend)
        backend = std::make_unique<NativeRenderSystem>(device, *this);
    initialize(std::move(backend), width, height);
    if (!_device) {
        _device = &device;
        _profiler.bind(device.profileContext(), &device);
    }
}

void Renderer::initialize(std::unique_ptr<RenderBackend> backend, int width,
                          int height) {
    if (_renderBackend || _device)
        throw std::logic_error("Renderer is already initialized or bound");
    if (!backend)
        throw std::invalid_argument("Render backend must not be null");
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("Render dimensions must be positive");
    _renderBackend = std::move(backend);
    try {
        _renderBackend->initialize(width, height);
        _backendCapabilities = _renderBackend->capabilities();
        _pickView.width = width;
        _pickView.height = height;
    } catch (...) {
        // Initialization may have created resources or bound compatibility
        // access.
        release();
        throw;
    }
}

RenderBackend& Renderer::renderBackend() {
    if (!_renderBackend)
        throw std::runtime_error("Renderer has no initialized render backend");
    return *_renderBackend;
}

NativeRenderSystem& Renderer::nativeRenderSystem() {
    auto* native = dynamic_cast<NativeRenderSystem*>(_renderBackend.get());
    if (!native)
        throw std::runtime_error(
            "This operation requires the Native render backend");
    return *native;
}

NativeRenderExtension& Renderer::native() {
    if (!_native)
        throw std::runtime_error(
            "This operation requires the Native render backend");
    return *_native;
}

const NativeRenderExtension& Renderer::native() const {
    if (!_native)
        throw std::runtime_error(
            "This operation requires the Native render backend");
    return *_native;
}

RenderBackendCapabilities Renderer::capabilities() const {
    return _backendCapabilities;
}

void Renderer::resizeFrame(int width, int height) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("Render dimensions must be positive");
    if (_renderBackend)
        _renderBackend->resize(width, height);
    cancelPick();
    _pickView.width = width;
    _pickView.height = height;
}

void Renderer::setBackgroundColor(const glm::vec4& color) {
    renderBackend().setBackgroundColor(color);
}

void Renderer::setGroundStyle(const BackgroundSettings& settings) {
    renderBackend().setGroundStyle(settings);
}

void Renderer::setExposure(float exposure) {
    if (!std::isfinite(exposure) || exposure < 0)
        throw std::invalid_argument("Exposure must be finite and nonnegative");
    renderBackend().setExposure(exposure);
}

void Renderer::syncFrame(Scene::SceneBackend* scene, const glm::mat4& view,
                         const glm::mat4& projection) {
    if (_pickView.view != view || _pickView.projection != projection)
        cancelPick();
    _pickView.view = view;
    _pickView.projection = projection;
    ++_pickView.frame;
    syncSceneLights(scene);
    syncSceneInstances();
    if (_renderBackend)
        _renderBackend->syncFrame(view, projection);
    else if (_rasterizer)
        _rasterizer->updateFrameData(view, projection);
}

void Renderer::syncSceneInstances() {
    for (auto& [key, batch] : _sceneBatches) {
        if (!_rasterizer && !batch->bridge.empty())
            batch->materialSource = Scene::SceneRenderBridge::materialSource(
                *batch->material, _sceneResources);
        batch->bridge.collect();
    }
}

void Renderer::bindSceneResources(
    const Scene::SceneResourceManager& resources) {
    if (!_sceneBatches.empty())
        throw std::logic_error(
            "Bind scene resources before registering batches");
    _sceneResources = &resources;
}

void Renderer::syncScene(const Scene::RenderSceneFrame& scene) {
    if (!capabilities().staticScene)
        throw std::runtime_error(
            "Render backend does not support static scene data");
    renderBackend().syncScene(scene);
    _snapshotLighting = true;
}

void Renderer::prepareFrame(Camera& camera, UpAxis upAxis) {
    renderBackend().prepareFrame(camera, upAxis);
}

void Renderer::beginFrame() { renderBackend().beginFrame(); }

uint64_t Renderer::requestPick(glm::vec2 ndc) {
    if (!capabilities().screenPicking)
        throw std::runtime_error("Render backend does not support screen picking");
    if (_pickView.frame == 0)
        throw std::logic_error("Synchronize the frame before requesting a pick");
    cancelPick();
    auto request = _pickView;
    request.id = _nextPickId++;
    request.ndc = ndc;
    for (auto& [key, batch] : _sceneBatches)
        batch->bridge.refreshSelection();
    renderBackend().requestPick(request);
    _pendingPick = request;
    _pickInstanceRevision = _instanceRevision;
    return request.id;
}

std::optional<RenderPickResult> Renderer::pollPick() {
    if (!_pendingPick)
        return std::nullopt;
    auto result = renderBackend().pollPick();
    if (!result || result->requestId != _pendingPick->id ||
        result->frame != _pendingPick->frame)
        return std::nullopt;
    cancelPick();
    // External rows and legacy registrations have no stable object identity.
    if (result->hit.hit && result->hit.sceneInstanceId == 0 &&
        _pickInstanceRevision != _instanceRevision)
        return std::nullopt;
    return result;
}

bool Renderer::buildPrimSelection(Scene::Prim& prim,
                                   RayPickResult& selection) const {
    if (_rasterizer)
        return native().rasterizer().buildPrimSelection(&prim, selection);
    for (const auto& [key, batch] : _sceneBatches) {
        if (std::get<2>(key) != TransformSource::SceneGraph)
            continue;
        const int index = batch->bridge.instanceIndex(prim);
        if (index < 0)
            continue;
        selection = {};
        selection.hit = true;
        selection.handle = batch->handle;
        selection.instanceIndex = index;
        selection.sceneInstanceId = batch->bridge.instances().ids[index];
        selection.prim = &prim;
        return true;
    }
    return false;
}

RayPickResult Renderer::resolvePick(const RenderPickHit& hit) const {
    if (!hit.hit)
        return {};
    const auto found = _sceneBatchHandles.find(hit.handle);
    if (found != _sceneBatchHandles.end() &&
        hit.transformSource == TransformSource::SceneGraph) {
        const auto& bridge = found->second->bridge;
        auto* prim = bridge.resolve(hit.sceneInstanceId);
        const int index = bridge.instanceIndex(hit.sceneInstanceId);
        if (!prim || index < 0)
            return {};
        RayPickResult result;
        static_cast<RenderPickHit&>(result) = hit;
        result.prim = prim;
        result.instanceIndex = index;
        return result;
    }
    // Existing direct Rasterizer registrations retain Native resolution.
    if (_rasterizer)
        return native().rasterizer().resolvePick(hit);
    if (found == _sceneBatchHandles.end() || found->second->bridge.empty())
        return {};
    RayPickResult result;
    static_cast<RenderPickHit&>(result) = hit;
    const auto& bridge = found->second->bridge;
    if (!bridge.instances().ids.empty())
        result.prim = bridge.resolve(bridge.instances().ids.front());
    return result;
}

void Renderer::renderScene(const glm::mat4& view, const glm::mat4& projection,
                           bool wireframe) {
    if (wireframe && !capabilities().wireframe)
        throw std::runtime_error("Render backend does not support wireframe");
    renderBackend().renderScene(view, projection, wireframe);
}

void Renderer::endFrame(const RayPickResult& selection, bool presentToScreen) {
    const auto supported = capabilities();
    if (selection.hit && !supported.selectionOutline)
        throw std::runtime_error(
            "Render backend does not support selection outlines");
    if (presentToScreen && !supported.screenPresentation)
        throw std::runtime_error(
            "Render backend does not support screen presentation");
    renderBackend().endFrame(selection, presentToScreen);
}

RenderImage Renderer::readImage(bool flipY) {
    return renderBackend().readImage(flipY);
}

RenderImage Renderer::readImageResized(int width, int height, bool flipY) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument(
            "Capture width and height must be positive");
    return renderBackend().readImageResized(width, height, flipY);
}

Backend::Texture* Renderer::presentedTexture() {
    auto* native = dynamic_cast<NativeRenderSystem*>(_renderBackend.get());
    auto* output = native ? native->outputFramebuffer() : nullptr;
    return output ? output->getColorTexture() : nullptr;
}

const CudaRenderImage* Renderer::cudaImage() const {
    return _renderBackend ? _renderBackend->cudaImage() : nullptr;
}

Backend::Texture* Renderer::renderCameraPreview(const glm::mat4& view,
                                                const glm::mat4& projection,
                                                int width, int height) {
    return nativeRenderSystem().renderCameraPreview(view, projection, width,
                                                    height);
}

RenderImage Renderer::readCameraPreview(bool flipY) {
    return nativeRenderSystem().readCameraPreview(flipY);
}

int Renderer::sampleCount() const {
    const auto* native =
        dynamic_cast<const NativeRenderSystem*>(_renderBackend.get());
    return native ? native->sampleCount() : 0;
}

void Renderer::rebuildSceneRenderTarget() {
    if (_renderBackend)
        nativeRenderSystem().rebuildSceneRenderTarget();
}

void Renderer::rebuildSceneClearTarget(const glm::vec4& clearColor) {
    if (_renderBackend)
        nativeRenderSystem().rebuildSceneClearTarget(clearColor);
}

void Renderer::release() {
    cancelPick();
    _pickView = {};
    _native.reset();
    _profiler.bind(nullptr);
    _profiler.setEnabled(false);
    _device = nullptr;
    _rasterizer = nullptr;
    _postProcessor = nullptr;
    _selectionOutlineProcessor = nullptr;
    _renderBackend.reset();
    _sceneBatchHandles.clear();
    _sceneBatches.clear();
    _sceneResources = nullptr;
    _backendCapabilities = {};
    _lightCollector = {};
    _snapshotLighting = false;
    _sceneLighting = false;
}

void Renderer::bind(Backend::GraphicsDevice* device, Rasterizer* rasterizer,
                    PostProcessor* postProcessor,
                    SelectionOutlineProcessor* selectionOutlineProcessor) {
    if (!device || !rasterizer)
        throw std::invalid_argument(
            "Native binding requires a device and Rasterizer");
    _native = std::make_unique<NativeRenderExtension>(*device, *rasterizer,
                                                      _settings);
    _device = device;
    _profiler.bind(device->profileContext(), device);
    _rasterizer = rasterizer;
    _postProcessor = postProcessor;
    _selectionOutlineProcessor = selectionOutlineProcessor;
}

void Renderer::setViewportSize(int width, int height) {
    native().setViewportSize(width, height);
}

void Renderer::applyBackgroundSettings() {
    native().rasterizer().setBackgroundSettings(_settings.background);
}

void Renderer::setLight(const DirectionalLight& light) {
    native().rasterizer().setLight(light);
}

const DirectionalLight& Renderer::light() const {
    if (!_rasterizer)
        throw std::runtime_error(
            "Renderer light access requires the Native backend");
    return _rasterizer->getLight();
}

void Renderer::setPointLights(std::vector<PointLight> lights) {
    native().rasterizer().setPointLights(std::move(lights));
}

const std::vector<PointLight>& Renderer::pointLights() const {
    if (!_rasterizer)
        throw std::runtime_error(
            "Renderer light access requires the Native backend");
    return _rasterizer->getPointLights();
}

void Renderer::setSpotLights(std::vector<SpotLight> lights) {
    native().rasterizer().setSpotLights(std::move(lights));
}

const std::vector<SpotLight>& Renderer::spotLights() const {
    if (!_rasterizer)
        throw std::runtime_error(
            "Renderer light access requires the Native backend");
    return _rasterizer->getSpotLights();
}

void Renderer::syncSceneLights(Scene::SceneBackend* scene) {
    if (_snapshotLighting ||
        (_renderBackend ? !capabilities().sceneBatches : !_rasterizer))
        return;
    auto lights =
        _lightCollector.collect(scene ? scene->getRootPrim() : nullptr);
    // Preserve explicit Native light setters until a scene takes ownership.
    if (!lights.authored && !_sceneLighting)
        return;
    if (_renderBackend)
        _renderBackend->syncLights(lights);
    else
        _rasterizer->setSceneLights(lights);
    _sceneLighting = lights.authored;
}

Backend::Framebuffer* Renderer::shadowFbo() {
    return _rasterizer ? _rasterizer->getShadowFbo() : nullptr;
}

RenderHookHandle Renderer::addRenderHook(RenderHookPhase phase,
                                         RenderHookCallback callback) {
    return native().rasterizer().addRenderHook(phase, std::move(callback));
}

bool Renderer::removeRenderHook(RenderHookHandle handle) {
    return native().rasterizer().removeRenderHook(handle);
}

std::unique_ptr<Backend::GraphicsPipeline>
Renderer::createSceneHookPipeline(const SceneHookPipelineDesc& desc) {
    return native().createSceneHookPipeline(desc);
}

void Renderer::renderSceneToFramebuffer(Camera& camera,
                                        Backend::Framebuffer* target, int width,
                                        int height, bool clear) {
    renderSceneToFramebuffer(camera.getViewMatrix(), camera.getProjMatrix(),
                             target, width, height, clear);
}

void Renderer::renderSceneToFramebuffer(const glm::mat4& view,
                                        const glm::mat4& projection,
                                        Backend::Framebuffer* target, int width,
                                        int height, bool clear) {
    auto& extension = native();
    if (!target || width <= 0 || height <= 0)
        return;
    applyBackgroundSettings();
    syncSceneInstances();
    if (_renderBackend)
        _renderBackend->syncFrame(view, projection);
    else
        extension.rasterizer().updateFrameData(view, projection);
    extension.renderSceneToFramebuffer(view, projection, target, width, height,
                                       clear);
}

RenderableHandle Renderer::addRenderable(Material* material, Scene::Prim* prim,
                                         TransformSource transformSource) {
    if (capabilities().sceneBatches ||
        (capabilities().cpuInstanceTransformsInput &&
         transformSource == TransformSource::ExternalBuffer)) {
        if (!prim || !material)
            return InvalidHandle;
        return addSceneRenderable(*material, *prim, transformSource, nullptr);
    }
    if (_renderBackend && !_rasterizer)
        throw std::runtime_error(
            "Render backend does not support Native renderable registration");
    return _rasterizer
               ? _rasterizer->addRenderable(material, prim, transformSource)
               : InvalidHandle;
}

RenderableHandle
Renderer::addSceneRenderable(Material& material, Scene::Prim& prim,
                             TransformSource source,
                             const Scene::SkinnedMeshData* skinnedMesh) {
    const auto mesh = Scene::SceneRenderBridge::meshData(prim);
    if (!mesh.mesh || mesh.mesh->vertices.empty() ||
        mesh.mesh->indices.empty() ||
        (skinnedMesh && !skinnedMesh->hasValidVertexSkinning()))
        return InvalidHandle;

    // Skinning matrices and deformable vertices belong to one registration.
    const size_t skinningId = skinnedMesh ? _nextSkinningId++ : 0;
    const auto key =
        std::make_tuple(mesh.mesh.get(), &material, source, skinningId);
    SceneBatchState state;
    if (auto component = prim.getRenderComponent())
        state = {component->isDoubleSided(), component->castsShadow(),
                 component->alphaMode(), component->alphaCutoff()};

    const auto [first, last] = _sceneBatches.equal_range(key);
    if (source == TransformSource::SceneGraph && !skinnedMesh) {
        for (auto candidate = first; candidate != last; ++candidate) {
            auto& batch = *candidate->second;
            if (!batch.hasGeometryUpdates || !batch.bridge.contains(prim))
                continue;
            if (batch.state != state && batch.bridge.size() > 1)
                throw std::runtime_error("cannot split render policies of a "
                                         "shared dynamic geometry batch");
            setSceneBatchState(batch.handle, state);
            return batch.handle;
        }
    }
    auto it = std::find_if(first, last, [&](const auto& entry) {
        return source == TransformSource::ExternalBuffer ||
               entry.second->state == state;
    });
    if (it == last && source == TransformSource::SceneGraph) {
        // A sole owner can change policy without reallocating GPU geometry.
        it = std::find_if(first, last, [&](const auto& entry) {
            return entry.second->bridge.size() == 1 &&
                   entry.second->bridge.contains(prim);
        });
        if (it != last) {
            setSceneBatchState(it->second->handle, state);
            return it->second->handle;
        }
    }

    if (it == last) {
        auto batch = std::make_unique<SceneBatch>();
        batch->key = key;
        batch->mesh = mesh;
        batch->material = &material;
        batch->state = state;

        // Native keeps live legacy material bindings; other backends receive
        // source values and textures only.
        batch->materialSource = Scene::SceneRenderBridge::materialSource(
            material, _rasterizer ? nullptr : _sceneResources);
        auto* native = dynamic_cast<NativeRenderSystem*>(_renderBackend.get());
        if (native)
            native->bindLegacyMaterial(batch->materialSource, &material);

        try {
            batch->handle = renderBackend().registerSceneBatch(
                mesh, batch->materialSource, batch->bridge.instances(), source,
                skinnedMesh);
        } catch (...) {
            if (native)
                native->bindLegacyMaterial(batch->materialSource, nullptr);
            throw;
        }
        if (batch->handle == InvalidHandle) {
            if (native)
                native->bindLegacyMaterial(batch->materialSource, nullptr);
            return InvalidHandle;
        }
        try {
            renderBackend().setSceneBatchState(batch->handle, state);
        } catch (...) {
            renderBackend().removeSceneBatch(batch->handle);
            throw;
        }
        it = _sceneBatches.emplace(key, std::move(batch));
        _sceneBatchHandles.emplace(it->second->handle, it->second.get());
    }

    auto& batch = *it->second;
    if (batch.bridge.contains(prim))
        return batch.handle;
    batch.bridge.add(prim);
    if (_rasterizer)
        _rasterizer->attachScenePrim(batch.handle, prim, batch.bridge);
    return batch.handle;
}

RenderableHandle
Renderer::addSkinnedRenderable(Material* material, Scene::Prim* prim,
                               const Scene::SkinnedMeshData& skinnedMesh,
                               TransformSource transformSource) {
    if (capabilities().sceneBatches || capabilities().skinning) {
        if (!prim || !material)
            return InvalidHandle;
        return addSceneRenderable(*material, *prim, transformSource,
                                  &skinnedMesh);
    }
    if (_renderBackend && !_rasterizer)
        throw std::runtime_error(
            "Render backend does not support Native renderable registration");
    return _rasterizer ? _rasterizer->addSkinnedRenderable(
                             material, prim, skinnedMesh, transformSource)
                       : InvalidHandle;
}

void Renderer::releaseEmptyBatch(RenderableHandle handle) {
    const auto found = _sceneBatchHandles.find(handle);
    if (found == _sceneBatchHandles.end())
        return;
    const auto& batch = *found->second;
    // External adapters release instance arrays with their last owner.
    const bool external = !_rasterizer && std::get<2>(batch.key) ==
                                              TransformSource::ExternalBuffer;
    if (!batch.bridge.empty() || (std::get<3>(batch.key) == 0 && !external))
        return;
    const auto key = batch.key;
    renderBackend().removeSceneBatch(handle);
    _sceneBatchHandles.erase(found);
    const auto [first, last] = _sceneBatches.equal_range(key);
    for (auto it = first; it != last; ++it) {
        if (it->second->handle == handle) {
            _sceneBatches.erase(it);
            break;
        }
    }
}

void Renderer::removePrim(RenderableHandle handle, Scene::Prim* prim) {
    ++_instanceRevision;
    if (_rasterizer)
        _rasterizer->removePrim(handle, prim);
    else if (prim) {
        auto it = _sceneBatchHandles.find(handle);
        if (it != _sceneBatchHandles.end())
            it->second->bridge.remove(*prim);
    }
    releaseEmptyBatch(handle);
}

void Renderer::removePrim(Scene::Prim* prim) {
    ++_instanceRevision;
    if (_rasterizer)
        _rasterizer->removePrim(prim);
    else if (prim) {
        for (auto& [key, batch] : _sceneBatches)
            batch->bridge.remove(*prim);
    }
    for (auto it = _sceneBatchHandles.begin();
         it != _sceneBatchHandles.end();) {
        const auto handle = it++->first;
        releaseEmptyBatch(handle);
    }
}

void Renderer::updateRenderableTransforms(
    RenderableHandle handle, const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    ++_instanceRevision;
    if (_renderBackend)
        _renderBackend->updateRenderableTransforms(handle, transforms, colors);
    else if (_rasterizer)
        _rasterizer->updateRenderableTransforms(handle, transforms, colors);
}

void Renderer::setRenderableExternalBuffer(RenderableHandle handle,
                                           const ExternalBufferDesc& desc) {
    ++_instanceRevision;
    if (_renderBackend)
        _renderBackend->setRenderableExternalBuffer(handle, desc);
    else if (_rasterizer)
        _rasterizer->setRenderableExternalBuffer(handle, desc);
}

std::vector<Sim::GpuArrayView> Renderer::mapRenderableCudaTransformBuffers(
    const std::vector<RenderableHandle>& handles, int count, int deviceId,
    uint64_t streamHandle) {
    return native().rasterizer().mapRenderableCudaTransformBuffers(
        handles, count, deviceId, streamHandle);
}

void Renderer::unmapRenderableCudaTransformBuffers(
    const std::vector<RenderableHandle>& handles, int deviceId,
    uint64_t streamHandle) {
    native().rasterizer().unmapRenderableCudaTransformBuffers(handles, deviceId,
                                                              streamHandle);
}

bool Renderer::getRenderableInstanceTransform(RenderableHandle handle,
                                              int instanceIndex,
                                              glm::mat4& outTransform) const {
    if (_renderBackend)
        return _renderBackend->getRenderableInstanceTransform(
            handle, instanceIndex, outTransform);
    return _rasterizer && _rasterizer->getRenderableInstanceTransform(
                              handle, instanceIndex, outTransform);
}

bool Renderer::setRenderableInstanceTransform(RenderableHandle handle,
                                              int instanceIndex,
                                              const glm::mat4& transform) {
    if (_renderBackend)
        return _renderBackend->setRenderableInstanceTransform(
            handle, instanceIndex, transform);
    return _rasterizer && _rasterizer->setRenderableInstanceTransform(
                              handle, instanceIndex, transform);
}

void Renderer::setRenderableColors(RenderableHandle handle,
                                   const std::vector<glm::vec4>& colors) {
    if (_renderBackend)
        _renderBackend->setRenderableColors(handle, colors);
    else if (_rasterizer)
        _rasterizer->setRenderableColors(handle, colors);
}

void Renderer::setRenderableDoubleSided(RenderableHandle handle,
                                        bool doubleSided) {
    auto batch = _sceneBatchHandles.find(handle);
    if (batch != _sceneBatchHandles.end()) {
        batch->second->state.doubleSided = doubleSided;
        renderBackend().setSceneBatchState(handle, batch->second->state);
        return;
    }
    if (_rasterizer)
        _rasterizer->setRenderableDoubleSided(handle, doubleSided);
}

void Renderer::setRenderableCastsShadow(RenderableHandle handle,
                                        bool castsShadow) {
    auto batch = _sceneBatchHandles.find(handle);
    if (batch != _sceneBatchHandles.end()) {
        batch->second->state.castsShadow = castsShadow;
        renderBackend().setSceneBatchState(handle, batch->second->state);
        return;
    }
    if (_rasterizer)
        _rasterizer->setRenderableCastsShadow(handle, castsShadow);
}

void Renderer::setSceneBatchState(RenderableHandle handle,
                                  const SceneBatchState& state) {
    auto batch = _sceneBatchHandles.find(handle);
    if (batch != _sceneBatchHandles.end()) {
        batch->second->state = state;
        renderBackend().setSceneBatchState(handle, state);
    } else if (_rasterizer) {
        _rasterizer->setRenderableDoubleSided(handle, state.doubleSided);
        _rasterizer->setRenderableCastsShadow(handle, state.castsShadow);
        _rasterizer->setRenderableAlphaMode(handle, state.alphaMode,
                                            state.alphaCutoff);
    }
}

void Renderer::setRenderableAlphaMode(RenderableHandle handle, AlphaMode mode,
                                      float cutoff) {
    auto batch = _sceneBatchHandles.find(handle);
    if (batch != _sceneBatchHandles.end()) {
        batch->second->state.alphaMode = mode;
        batch->second->state.alphaCutoff = cutoff;
        renderBackend().setSceneBatchState(handle, batch->second->state);
        return;
    }
    if (_rasterizer)
        _rasterizer->setRenderableAlphaMode(handle, mode, cutoff);
}

void Renderer::setRenderableTexture(RenderableHandle handle,
                                    Backend::Texture* tex, TextureRole role) {
    setRenderableTexture(handle, tex, textureRoleSlot(role));
}

void Renderer::setRenderableTexture(RenderableHandle handle,
                                    Backend::Texture* tex, int slot) {
    native().rasterizer().setRenderableTexture(handle, tex, slot);
}

void Renderer::updateRenderableGeometry(RenderableHandle handle,
                                        const std::vector<glm::vec3>& positions,
                                        const std::vector<glm::vec3>& normals) {
    if (_renderBackend)
        _renderBackend->updateRenderableGeometry(handle, positions, normals);
    else if (_rasterizer)
        _rasterizer->updateRenderableGeometry(handle, positions, normals);
    if (auto it = _sceneBatchHandles.find(handle);
        it != _sceneBatchHandles.end())
        it->second->hasGeometryUpdates = true;
}

void Renderer::updateRenderableSkinningMatrices(
    RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices) {
    if (_renderBackend)
        _renderBackend->updateRenderableSkinningMatrices(handle, boneMatrices);
    else if (_rasterizer)
        _rasterizer->updateRenderableSkinningMatrices(handle, boneMatrices);
}

} // namespace KE

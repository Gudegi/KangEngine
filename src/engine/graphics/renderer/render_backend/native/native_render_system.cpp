#include "native_render_system.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/graphics/renderer/selection_outline_processor.hpp"
#include "engine/graphics/renderer/render_backend/native/native_render_extension.hpp"
#include "engine/scene/scene_resource_manager.hpp"

#include <stdexcept>
#include <set>
#include <filesystem>

namespace KE {
namespace {

RenderImage readFramebuffer(Backend::Framebuffer* framebuffer, bool flipY) {
    if (!framebuffer)
        throw std::runtime_error("No completed scene frame is available");
    const auto* color = framebuffer->getColorTexture();
    return {color->getWidth(), color->getHeight(),
            framebuffer->readColorPixels(flipY)};
}

} // namespace

NativeRenderSystem::NativeRenderSystem(Backend::GraphicsDevice& device,
                                       Renderer& renderer)
    : _device(device), _renderer(renderer) {}

NativeRenderSystem::~NativeRenderSystem() = default;

void NativeRenderSystem::initialize(int width, int height) {
    _width = width;
    _height = height;
    Backend::FramebufferDesc sceneFramebufferDesc;
    sceneFramebufferDesc.width = _width;
    sceneFramebufferDesc.height = _height;
    sceneFramebufferDesc.stencil = true;
    // Resolve once after all scene draws, before post-processing and readback.
    sceneFramebufferDesc.msaaSamples = 0;
    sceneFramebufferDesc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
    _framebuffer = _device.createFramebuffer(sceneFramebufferDesc);
    rebuildSceneRenderTarget();
    _selectionMaskFramebuffer =
        _device.createFramebuffer({_width, _height, false, false, 0});

    _rasterizer = std::make_unique<Rasterizer>(&_device);
    _postProcessor = std::make_unique<PostProcessor>();
    _postProcessor->init(&_device, _width, _height);
    _previewPostProcessor = std::make_unique<PostProcessor>();
    _previewPostProcessor->init(&_device, _width, _height);
    _previewPostWidth = _width;
    _previewPostHeight = _height;
    _selectionOutlineProcessor = std::make_unique<SelectionOutlineProcessor>();
    _selectionOutlineProcessor->init(&_device, _width, _height);
    _renderer.bind(&_device, _rasterizer.get(), _postProcessor.get(),
                   _selectionOutlineProcessor.get());
    _renderer.setViewportSize(_width, _height);
}

void NativeRenderSystem::resize(int width, int height) {
    _output = nullptr;
    _width = width;
    _height = height;
    _framebuffer->resize(width, height);
    rebuildSceneRenderTarget();
    _selectionMaskFramebuffer->resize(width, height);
    _postProcessor->resize(width, height);
    _previewPostProcessor->resize(width, height);
    _previewPostWidth = width;
    _previewPostHeight = height;
    _selectionOutlineProcessor->resize(width, height);
    _renderer.setViewportSize(width, height);
}

void NativeRenderSystem::prepareFrame(Camera& camera, UpAxis upAxis) {
    _rasterizer->renderShadowMap(camera, upAxis, _width, _height);
}

void NativeRenderSystem::syncFrame(const glm::mat4& view,
                                   const glm::mat4& projection) {
    for (auto& [source, material] : _sourceMaterials)
        updateMaterialSource(*source, *material);
    _rasterizer->updateFrameData(view, projection);
}

void NativeRenderSystem::bindLegacyMaterial(
    const Scene::RenderMaterialSource& source, Material* material) {
    if (material)
        _legacyMaterials[&source] = material;
    else
        _legacyMaterials.erase(&source);
}

RenderableHandle NativeRenderSystem::registerSceneBatch(
    const Scene::RenderMeshData& mesh,
    const Scene::RenderMaterialSource& material,
    const Scene::RenderInstanceData& instances, TransformSource source,
    const Scene::SkinnedMeshData* skinnedMesh) {
    auto legacy = _legacyMaterials.find(&material);
    Material* native =
        legacy == _legacyMaterials.end() ? nullptr : legacy->second;
    bool createdSource = false;
    if (!native) {
        auto it = _sourceMaterials.find(&material);
        if (it == _sourceMaterials.end()) {
            std::unique_ptr<Material> created;
            switch (material.values.shadingModel) {
            case MaterialShadingModel::Color:
                created = std::make_unique<ColorMaterial>();
                break;
            case MaterialShadingModel::Phong:
                created = std::make_unique<PhongMaterial>();
                break;
            case MaterialShadingModel::PBR:
                created = std::make_unique<PBRMaterial>();
                break;
            default:
                throw std::invalid_argument(
                    "Native backend does not support this material model");
            }
            updateMaterialSource(material, *created);
            it = _sourceMaterials.emplace(&material, std::move(created)).first;
            createdSource = true;
        }
        native = it->second.get();
    }
    try {
        const auto handle = _rasterizer->registerSceneBatch(
            mesh, native, instances, skinnedMesh, source);
        if (handle == InvalidHandle && createdSource)
            _sourceMaterials.erase(&material);
        return handle;
    } catch (...) {
        if (createdSource)
            _sourceMaterials.erase(&material);
        throw;
    }
}

void NativeRenderSystem::updateMaterialSource(
    const Scene::RenderMaterialSource& source, Material& material) {
    const auto& value = source.values;
    if (value.hasNativeTextures)
        throw std::invalid_argument(
            "Material has unresolved Native texture bindings");
    if (value.shadingModel != material.shadingModel())
        throw std::invalid_argument(
            "Register a new material source to change its shading model");

    auto texture = [&](const char* slot) -> Backend::Texture* {
        const auto entry = source.textures.find(slot);
        if (entry == source.textures.end())
            return nullptr;
        const auto& definition = entry->second.second;
        if (!definition)
            throw std::invalid_argument("Texture source is missing");
        auto found = _materialTextures.find(definition);
        if (found == _materialTextures.end()) {
            auto image = _renderer.native().createTexture(*definition);
            if (!image)
                throw std::runtime_error("Failed to create material texture: " +
                                         definition->path);
            found =
                _materialTextures.emplace(definition, std::move(image)).first;
        }
        return found->second.get();
    };
    if (auto* color = dynamic_cast<ColorMaterial*>(&material)) {
        if (!source.textures.empty())
            throw std::invalid_argument(
                "Color material textures require Native batch bindings");
        color->setStyle(value.colorStyle);
    } else if (auto* phong = dynamic_cast<PhongMaterial*>(&material)) {
        for (const auto& [slot, resource] : source.textures)
            if (slot != "diffuse_map" && slot != "specular_map" &&
                slot != "alpha_map" && slot != "normal_map")
                throw std::invalid_argument("Unsupported Phong texture slot: " +
                                            slot);
        phong->ambient = value.ambient;
        phong->diffuse = glm::vec3(value.baseColor);
        phong->specular = value.specular;
        phong->shininess = value.shininess;
        phong->diffuseMap = texture("diffuse_map");
        phong->specularMap = texture("specular_map");
        phong->alphaMap = texture("alpha_map");
        phong->normalMap = texture("normal_map");
    } else if (auto* pbr = dynamic_cast<PBRMaterial*>(&material)) {
        for (const auto& [slot, resource] : source.textures)
            if (slot != "base_color_texture" && slot != "normal_texture" &&
                slot != "metallic_roughness_texture" &&
                slot != "metallic_texture" && slot != "roughness_texture" &&
                slot != "ao_texture" && slot != "orm_texture" &&
                slot != "emissive_texture" && slot != "opacity_texture")
                throw std::invalid_argument("Unsupported PBR texture slot: " +
                                            slot);
        pbr->baseColor = value.baseColor;
        pbr->metallic = value.metallic;
        pbr->roughness = value.roughness;
        pbr->emissiveColor = value.emissiveColor;
        pbr->emissiveStrength = value.emissiveStrength;
        pbr->baseColorTexture = texture("base_color_texture");
        pbr->normalTexture = texture("normal_texture");
        pbr->metallicRoughnessTexture = texture("metallic_roughness_texture");
        pbr->metallicTexture = texture("metallic_texture");
        pbr->roughnessTexture = texture("roughness_texture");
        pbr->aoTexture = texture("ao_texture");
        pbr->ormTexture = texture("orm_texture");
        pbr->emissiveTexture = texture("emissive_texture");
        pbr->opacityTexture = texture("opacity_texture");
    }
    material.setUnlit(value.unlit);
}

void NativeRenderSystem::setSceneBatchState(RenderableHandle handle,
                                            const SceneBatchState& state) {
    _rasterizer->setRenderableDoubleSided(handle, state.doubleSided);
    _rasterizer->setRenderableCastsShadow(handle, state.castsShadow);
    _rasterizer->setRenderableAlphaMode(handle, state.alphaMode,
                                        state.alphaCutoff);
}

void NativeRenderSystem::updateRenderableTransforms(
    RenderableHandle handle, const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    _rasterizer->updateRenderableTransforms(handle, transforms, colors);
}

void NativeRenderSystem::setRenderableExternalBuffer(
    RenderableHandle handle, const ExternalBufferDesc& desc) {
    _rasterizer->setRenderableExternalBuffer(handle, desc);
}

bool NativeRenderSystem::getRenderableInstanceTransform(
    RenderableHandle handle, int instanceIndex, glm::mat4& outTransform) const {
    return _rasterizer->getRenderableInstanceTransform(handle, instanceIndex,
                                                       outTransform);
}

bool NativeRenderSystem::setRenderableInstanceTransform(
    RenderableHandle handle, int instanceIndex, const glm::mat4& transform) {
    return _rasterizer->setRenderableInstanceTransform(handle, instanceIndex,
                                                       transform);
}

void NativeRenderSystem::setRenderableColors(
    RenderableHandle handle, const std::vector<glm::vec4>& colors) {
    _rasterizer->setRenderableColors(handle, colors);
}

void NativeRenderSystem::updateRenderableGeometry(
    RenderableHandle handle, const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals) {
    _rasterizer->updateRenderableGeometry(handle, positions, normals);
}

void NativeRenderSystem::updateRenderableSkinningMatrices(
    RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices) {
    _rasterizer->updateRenderableSkinningMatrices(handle, boneMatrices);
}

void NativeRenderSystem::syncScene(const Scene::RenderSceneFrame& scene) {
    if (scene.incremental && scene.objects.empty() &&
        scene.removedObjects.empty()) {
        if (scene.lightingChanged) {
            syncLights(scene.lighting);
            _sceneLighting = scene.lighting;
        }
        return;
    }
    std::set<Scene::RenderInstanceId> ids;
    for (const auto& object : scene.objects) {
        if (!object.id || !ids.insert(object.id).second)
            throw std::invalid_argument(
                "Scene update requires unique nonzero instance IDs");
    }
    auto objects =
        scene.incremental ? _sceneObjects : decltype(_sceneObjects){};
    for (auto id : scene.removedObjects)
        objects.erase(id);
    for (const auto& object : scene.objects)
        objects[object.id] = object;
    Scene::RenderSceneFrame snapshot;
    snapshot.lighting = scene.lightingChanged ? scene.lighting : _sceneLighting;
    snapshot.objects.reserve(objects.size());
    for (const auto& [id, object] : objects)
        snapshot.objects.push_back(object);
    syncSceneSnapshot(snapshot);
    _sceneObjects = std::move(objects);
    _sceneLighting = std::move(snapshot.lighting);
}

void NativeRenderSystem::syncSceneSnapshot(
    const Scene::RenderSceneFrame& scene) {
    // Reject unsupported inputs before changing the last accepted scene.
    size_t rectangleCount = 0;
    for (const auto& source : scene.lighting.lights)
        rectangleCount +=
            std::holds_alternative<Scene::RectLight>(source.light);
    if (rectangleCount > MaxRectLights)
        throw std::invalid_argument(
            "Native supports at most 16 visible RectLights");
    std::set<Scene::ResourceHandle> textureIds;
    for (const auto& object : scene.objects) {
        if (object.mesh.resource == Scene::InvalidResourceHandle ||
            object.materialResource == Scene::InvalidResourceHandle ||
            !object.mesh.mesh || object.mesh.mesh->vertices.empty() ||
            object.mesh.mesh->indices.empty())
            throw std::invalid_argument(
                "Static scene requires registered triangle meshes");
        const auto& material = object.material;
        if (material.shadingModel != MaterialShadingModel::PBR ||
            material.unlit || material.hasNativeTextures ||
            material.baseColor.a != 1.0f || object.color.a != 1.0f ||
            !object.mesh.mesh->colors.empty())
            throw std::invalid_argument(
                "Static scene requires opaque lit PBR and source textures");
        const SceneBatchKey key{object.mesh.resource, object.materialResource,
                                object.doubleSided};
        auto existing = _sceneBatches.find(key);
        if (existing != _sceneBatches.end() &&
            existing->second->mesh.mesh != object.mesh.mesh)
            throw std::invalid_argument(
                "Register a new resource to replace static geometry");
        for (const auto& [slot, resource] : object.textures) {
            if (slot != "base_color_texture" ||
                resource.first == Scene::InvalidResourceHandle ||
                !resource.second)
                throw std::invalid_argument(
                    "Static scene supports base-color texture sources only");
            textureIds.insert(resource.first);
            if (!_sceneTextures.count(resource.first) &&
                !resource.second->path.empty() &&
                !std::filesystem::is_regular_file(resource.second->path))
                throw std::invalid_argument(
                    "Texture source file does not exist: " +
                    resource.second->path);
        }
    }

    // Decode/upload new images once; retained source identities reuse GPU
    // state.
    for (const auto& object : scene.objects) {
        for (const auto& [slot, resource] : object.textures) {
            if (_sceneTextures.count(resource.first))
                continue;
            const auto& source = *resource.second;
            auto texture = _renderer.native().createTexture(source);
            if (!texture)
                throw std::runtime_error("Failed to create scene texture: " +
                                         source.path);
            _sceneTextures.emplace(resource.first, std::move(texture));
        }
    }
    for (auto& [key, batch] : _sceneBatches) {
        batch->instances.transforms.clear();
        batch->instances.colors.clear();
        batch->instances.ids.clear();
        batch->instances.pickable.clear();
    }

    std::set<SceneBatchKey> liveBatches;
    for (const auto& object : scene.objects) {
        const SceneBatchKey key{object.mesh.resource, object.materialResource,
                                object.doubleSided};
        liveBatches.insert(key);
        auto& entry = _sceneBatches[key];
        if (!entry) {
            entry = std::make_unique<SceneBatch>();
            entry->mesh = object.mesh;
        }
        auto& batch = *entry;
        const auto& value = object.material;
        batch.material.baseColor = value.baseColor;
        batch.material.metallic = value.metallic;
        batch.material.roughness = value.roughness;
        batch.material.emissiveColor = value.emissiveColor;
        batch.material.emissiveStrength = value.emissiveStrength;
        auto texture = object.textures.find("base_color_texture");
        auto* baseColorTexture =
            texture == object.textures.end()
                ? nullptr
                : _sceneTextures.at(texture->second.first).get();
        if (batch.material.baseColorTexture != baseColorTexture) {
            // Hidden batches also retain bindings to textures being released.
            _rasterizer->invalidateSceneBatch(batch.handle);
            batch.material.baseColorTexture = baseColorTexture;
        }
        if (batch.handle == InvalidHandle) {
            batch.materialSource.values = value;
            bindLegacyMaterial(batch.materialSource, &batch.material);
            batch.handle = registerSceneBatch(batch.mesh, batch.materialSource,
                                              batch.instances);
            if (batch.handle == InvalidHandle)
                throw std::runtime_error(
                    "Failed to register static scene batch");
            setSceneBatchState(batch.handle, {object.doubleSided, true,
                                              AlphaMode::Opaque, 0.5f});
        }
        if (object.visible) {
            batch.instances.transforms.push_back(object.world);
            batch.instances.colors.push_back(object.color);
            batch.instances.ids.push_back(object.id);
            batch.instances.pickable.push_back(object.pickable);
        }
    }

    // Remove GPU borrowers before releasing their source
    // arrays/materials/images.
    for (auto it = _sceneBatches.begin(); it != _sceneBatches.end();) {
        if (liveBatches.count(it->first)) {
            ++it;
            continue;
        }
        auto& batch = *it->second;
        _rasterizer->removeSceneBatch(batch.mesh, &batch.material);
        bindLegacyMaterial(batch.materialSource, nullptr);
        it = _sceneBatches.erase(it);
    }
    for (auto it = _sceneTextures.begin(); it != _sceneTextures.end();) {
        if (textureIds.count(it->first))
            ++it;
        else
            it = _sceneTextures.erase(it);
    }
    syncLights(scene.lighting);
}

void NativeRenderSystem::syncLights(const Scene::RenderLightData& lighting) {
    _rasterizer->setSceneLights(lighting);
}

void NativeRenderSystem::beginFrame() {
    const glm::vec4& color = _renderer.settings().background.backgroundColor;
    rebuildSceneClearTarget(color);
    auto encoder = _device.createCommandEncoder();
    auto pass = encoder->beginRenderPass(_sceneClearTarget.get(),
                                         _device.profilePass("render/clear"));
    pass->end();
    auto commands = encoder->finish();
    _device.submit(*commands);
}

void NativeRenderSystem::renderScene(const glm::mat4& view,
                                     const glm::mat4& projection,
                                     bool wireframe) {
    _rasterizer->setWireframeEnabled(wireframe);
    _renderer.applyBackgroundSettings();
    _rasterizer->render(view, projection, _sceneDrawTarget.get());
}

void NativeRenderSystem::requestPick(const RenderPickRequest& request) {
    _pickResult.reset();
    const auto hit =
        _rasterizer->pickScreen(request.ndc, request.view, request.projection,
                                request.width, request.height);
    _pickResult = RenderPickResult{request.id, request.frame, hit};
}

std::optional<RenderPickResult> NativeRenderSystem::pollPick() {
    auto result = _pickResult;
    _pickResult.reset();
    return result;
}

void NativeRenderSystem::endFrame(const RenderPickHit& selection,
                                  bool presentToScreen) {
    {
        auto encoder = _device.createCommandEncoder();
        auto pass = encoder->beginRenderPass(
            _sceneRenderTarget.get(),
            _device.profilePass("render/final_resolve"));
        pass->end();
        auto commands = encoder->finish();
        _device.submit(*commands);
    }

    Backend::Texture* finalSource = _framebuffer->getColorTexture();

    if (_selectionOutlineProcessor->config().enabled && selection.hit &&
        selection.handle != InvalidHandle && selection.instanceIndex >= 0) {
        _rasterizer->renderSelectionMaskPass(
            selection, _selectionMaskFramebuffer.get(), _width, _height);
        _selectionOutlineProcessor->renderOutlineCompositePass(
            _framebuffer->getColorTexture(),
            _selectionMaskFramebuffer->getColorTexture());
        finalSource = _selectionOutlineProcessor->getResult();
    }

    auto postScope = _device.profileContext()->cpuScope("render/post");
    const auto& settings = _renderer.settings();
    _postProcessor->process(finalSource, settings.gamma, settings.toneMapMode,
                            settings.toneMapExposure, settings.bloom);
    _output = _postProcessor->getOutputFramebuffer();
    if (presentToScreen)
        present();
}

void NativeRenderSystem::present() {
    auto cpuBlit = _device.profileContext()->cpuScope("render/native_blit");
    auto gpuBlit = _device.profileExternalScope("render/native_blit");
    _postProcessor->blitToScreen(_width, _height);
}

Backend::Texture* NativeRenderSystem::renderCameraPreview(
    const glm::mat4& view, const glm::mat4& projection, int width, int height) {
    if (!_previewFramebuffer) {
        Backend::FramebufferDesc desc;
        desc.width = width;
        desc.height = height;
        desc.stencil = true;
        desc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
        _previewFramebuffer = _device.createFramebuffer(desc);
    } else if (Backend::Texture* color =
                   _previewFramebuffer->getColorTexture()) {
        if (color->getWidth() != width || color->getHeight() != height)
            _previewFramebuffer->resize(width, height);
    }

    const RendererSettings& settings = _renderer.settings();

    _renderer.renderSceneToFramebuffer(
        view, projection, _previewFramebuffer.get(), width, height, true);
    if (_previewPostWidth != width || _previewPostHeight != height) {
        _previewPostProcessor->resize(width, height);
        _previewPostWidth = width;
        _previewPostHeight = height;
    }
    _previewPostProcessor->process(_previewFramebuffer->getColorTexture(),
                                   settings.gamma, settings.toneMapMode,
                                   settings.toneMapExposure, settings.bloom);
    _device.setViewport(0, 0, _width, _height);
    return _previewPostProcessor->getResult();
}

Backend::Framebuffer* NativeRenderSystem::previewFramebuffer() const {
    return _previewPostProcessor->getOutputFramebuffer();
}

RenderImage NativeRenderSystem::readImage(bool flipY) {
    return readFramebuffer(_output, flipY);
}

RenderImage NativeRenderSystem::readImageResized(int width, int height,
                                                 bool flipY) {
    if (!_output)
        throw std::runtime_error("No completed scene frame is available");
    return {width, height,
            _output->readColorPixelsResized(width, height, flipY)};
}

RenderImage NativeRenderSystem::readCameraPreview(bool flipY) {
    return readFramebuffer(_previewFramebuffer ? previewFramebuffer() : nullptr,
                           flipY);
}

int NativeRenderSystem::sampleCount() const {
    return _sceneRenderTarget->getSampleCount();
}

void NativeRenderSystem::rebuildSceneRenderTarget() {
    if (!_framebuffer || _width <= 0 || _height <= 0)
        return;

    _sceneRenderTarget.reset();
    _sceneDrawTarget.reset();
    _sceneClearTarget.reset();
    _sceneMsaaDepthStencilView.reset();
    _sceneResolveColorView.reset();
    _sceneMsaaColorView.reset();
    _sceneMsaaDepthStencil.reset();
    _sceneMsaaColor.reset();

    Backend::TextureResourceDesc colorDesc;
    colorDesc.extent = {static_cast<uint32_t>(_width),
                        static_cast<uint32_t>(_height), 1};
    colorDesc.format = Backend::TextureFormat::RGBA16Float;
    colorDesc.usage = Backend::TextureUsage::RenderAttachment;
    colorDesc.sampleCount = 4;
    colorDesc.label = "main_scene_msaa_color";
    _sceneMsaaColor = _device.createTexture(colorDesc);

    Backend::TextureResourceDesc depthDesc;
    depthDesc.extent = colorDesc.extent;
    depthDesc.format = Backend::TextureFormat::Depth24Stencil8;
    depthDesc.usage = Backend::TextureUsage::RenderAttachment;
    depthDesc.sampleCount = 4;
    depthDesc.label = "main_scene_msaa_depth_stencil";
    _sceneMsaaDepthStencil = _device.createTexture(depthDesc);

    Backend::TextureViewDesc colorViewDesc;
    colorViewDesc.format = Backend::TextureFormat::RGBA16Float;
    colorViewDesc.label = "main_scene_msaa_color_view";
    _sceneMsaaColorView =
        _device.createTextureView(_sceneMsaaColor.get(), colorViewDesc);

    Backend::TextureViewDesc resolveViewDesc;
    resolveViewDesc.format = Backend::TextureFormat::RGBA16Float;
    resolveViewDesc.label = "main_scene_resolve_color_view";
    _sceneResolveColorView = _device.createTextureView(
        _framebuffer->getColorTexture(), resolveViewDesc);

    Backend::TextureViewDesc depthViewDesc;
    depthViewDesc.format = Backend::TextureFormat::Depth24Stencil8;
    depthViewDesc.aspect = Backend::TextureAspect::All;
    depthViewDesc.label = "main_scene_msaa_depth_stencil_view";
    _sceneMsaaDepthStencilView =
        _device.createTextureView(_sceneMsaaDepthStencil.get(), depthViewDesc);

    Backend::RenderPassDesc passDesc;
    passDesc.label = "main_scene_msaa_pass";
    passDesc.colorAttachments = {{_sceneMsaaColorView.get(),
                                  _sceneResolveColorView.get(),
                                  Backend::LoadOp::Load,
                                  Backend::StoreOp::Store,
                                  {}}};
    passDesc.depthStencilAttachment =
        Backend::DepthStencilAttachmentDesc{_sceneMsaaDepthStencilView.get(),
                                            Backend::LoadOp::Load,
                                            Backend::StoreOp::Store,
                                            1.0f,
                                            Backend::LoadOp::Load,
                                            Backend::StoreOp::Store,
                                            0};
    _sceneRenderTarget = _device.createRenderTarget(passDesc);

    passDesc.label = "main_scene_msaa_draw_pass";
    passDesc.colorAttachments[0].resolveTarget = nullptr;
    _sceneDrawTarget = _device.createRenderTarget(passDesc);
}

void NativeRenderSystem::rebuildSceneClearTarget(const glm::vec4& clearColor) {
    if (!_sceneMsaaColorView || !_sceneMsaaDepthStencilView)
        return;
    if (_sceneClearTarget) {
        const auto& current =
            _sceneClearTarget->getDesc().colorAttachments.front().clearValue;
        if (current.r == clearColor.r && current.g == clearColor.g &&
            current.b == clearColor.b && current.a == clearColor.a)
            return;
    }

    Backend::RenderPassDesc passDesc;
    passDesc.label = "main_scene_clear_pass";
    passDesc.colorAttachments = {
        {_sceneMsaaColorView.get(),
         nullptr,
         Backend::LoadOp::Clear,
         Backend::StoreOp::Store,
         {clearColor.r, clearColor.g, clearColor.b, clearColor.a}}};
    passDesc.depthStencilAttachment =
        Backend::DepthStencilAttachmentDesc{_sceneMsaaDepthStencilView.get(),
                                            Backend::LoadOp::Clear,
                                            Backend::StoreOp::Store,
                                            1.0f,
                                            Backend::LoadOp::Clear,
                                            Backend::StoreOp::Store,
                                            0};
    _sceneClearTarget = _device.createRenderTarget(passDesc);
}

} // namespace KE

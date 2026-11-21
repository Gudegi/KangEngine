// Checks backend selection, failure cleanup, and App callbacks without Native
// draws.
#include "engine/core/app/app.hpp"
#include "bridge/skin_visual_bridge.hpp"
#include "engine/graphics/renderer/render_backend/render_backend.hpp"
#include "engine/scene/render_scene_data_collector.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/scene/native/native_scene.hpp"

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

using namespace KE;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class F> void requireRejected(F operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "unsupported operation was accepted");
}

class RecordingBackend final : public RenderBackend {
  public:
    RecordingBackend(std::vector<std::string>& calls, bool fail = false,
                     bool staticScene = false, bool sceneBatches = false,
                     bool picking = false)
        : _calls(calls), _fail(fail), _staticScene(staticScene),
          _sceneBatches(sceneBatches), _picking(picking) {}
    ~RecordingBackend() override { _calls.push_back("destroy"); }
    RenderBackendCapabilities capabilities() const override {
        return {false,         _picking, false,   _staticScene,
                _sceneBatches, false,    _picking};
    }
    void initialize(int width, int height) override {
        _calls.push_back("initialize");
        if (_fail)
            throw std::runtime_error("injected initialization failure");
        _width = width;
        _height = height;
    }
    void resize(int width, int height) override {
        _calls.push_back("resize");
        _width = width;
        _height = height;
        _image = {};
    }
    void syncScene(const Scene::RenderSceneFrame& scene) override {
        _calls.push_back("scene_data");
        snapshot = scene;
    }
    Scene::RenderSceneFrame snapshot;
    void syncLights(const Scene::RenderLightData& lights) override {
        lighting = lights;
    }
    Scene::RenderLightData lighting;
    struct Batch {
        Scene::RenderMeshData mesh;
        const Scene::RenderMaterialSource* material;
        const Scene::RenderInstanceData* source;
        Scene::RenderInstanceData instances;
        SceneBatchState state;
        bool external = false;
        std::vector<glm::vec4> boneWeights;
        std::vector<glm::mat4> boneMatrices;
    };
    std::map<RenderableHandle, Batch> batches;
    RenderableHandle
    registerSceneBatch(const Scene::RenderMeshData& mesh,
                       const Scene::RenderMaterialSource& material,
                       const Scene::RenderInstanceData& instances,
                       TransformSource source,
                       const Scene::SkinnedMeshData* skin) override {
        require(_sceneBatches, "backend does not support scene batches");
        const auto handle = static_cast<RenderableHandle>(batches.size());
        batches.emplace(handle, Batch{mesh, &material, &instances, {}, {}});
        batches.at(handle).external = source == TransformSource::ExternalBuffer;
        if (skin)
            batches.at(handle).boneWeights = skin->boneWeights;
        return handle;
    }
    void setSceneBatchState(RenderableHandle handle,
                            const SceneBatchState& state) override {
        batches.at(handle).state = state;
    }
    void
    updateRenderableTransforms(RenderableHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors) override {
        if (!_sceneBatches)
            return RenderBackend::updateRenderableTransforms(handle, transforms,
                                                             colors);
        auto& batch = batches.at(handle);
        require(batch.external, "instance updates require external ownership");
        batch.instances.transforms = transforms;
        if (colors)
            batch.instances.colors = *colors;
    }
    const std::vector<glm::vec3>* lastPositions = nullptr;
    void
    updateRenderableGeometry(RenderableHandle handle,
                             const std::vector<glm::vec3>& positions,
                             const std::vector<glm::vec3>& normals) override {
        if (!_sceneBatches)
            return RenderBackend::updateRenderableGeometry(handle, positions,
                                                           normals);
        batches.at(handle);
        lastPositions = &positions;
    }
    void updateRenderableSkinningMatrices(
        RenderableHandle handle,
        const std::vector<glm::mat4>& matrices) override {
        if (!_sceneBatches)
            return RenderBackend::updateRenderableSkinningMatrices(handle,
                                                                   matrices);
        batches.at(handle).boneMatrices = matrices;
    }
    void syncFrame(const glm::mat4&, const glm::mat4&) override {
        _calls.push_back("sync");
        for (auto& [handle, batch] : batches) {
            if (!batch.external)
                batch.instances = *batch.source;
        }
    }
    void prepareFrame(Camera&, UpAxis) override { _calls.push_back("prepare"); }
    void beginFrame() override { _calls.push_back("begin"); }
    void renderScene(const glm::mat4&, const glm::mat4&, bool) override {
        _calls.push_back("scene");
    }
    RenderPickRequest pickRequest;
    std::optional<RenderPickResult> pickResponse;
    RenderPickHit selection;
    void requestPick(const RenderPickRequest& request) override {
        pickRequest = request;
    }
    std::optional<RenderPickResult> pollPick() override {
        auto result = pickResponse;
        pickResponse.reset();
        return result;
    }
    void endFrame(const RenderPickHit& selected,
                  bool presentToScreen) override {
        selection = selected;
        require(!presentToScreen, "App requested unsupported presentation");
        _calls.push_back("end");
        _image = {_width, _height,
                  std::vector<uint8_t>(
                      static_cast<size_t>(_width * _height * 3), 37)};
    }
    RenderImage readImage(bool) override {
        if (_image.pixels.empty())
            throw std::runtime_error("No completed frame");
        return _image;
    }
    RenderImage readImageResized(int width, int height, bool) override {
        readImage(true);
        return {
            width, height,
            std::vector<uint8_t>(static_cast<size_t>(width * height * 3), 37)};
    }

  private:
    std::vector<std::string>& _calls;
    bool _fail;
    bool _staticScene;
    bool _sceneBatches;
    bool _picking;
    int _width = 0;
    int _height = 0;
    RenderImage _image;
};

class ProbeApp final : public App {
  public:
    explicit ProbeApp(std::vector<std::string>& calls) : _calls(calls) {}
    void preUpdate() override { _calls.push_back("update"); }
    void fixedUpdate(double dt) override {
        require(dt > 0, "fixed update lost its time interval");
        _calls.push_back("fixed");
    }
    void preRender() override { _calls.push_back("pre"); }
    void render() override { _calls.push_back("custom"); }
    void postRender() override { _calls.push_back("post"); }
    std::vector<RayPickResult> picks;
    void onRayPicked(const RayPickResult& hit) override {
        picks.push_back(hit);
    }

  private:
    std::vector<std::string>& _calls;
};

void checkRenderer() {
    std::vector<std::string> calls;
    Renderer renderer;
    requireRejected([&] {
        renderer.initialize(std::make_unique<RecordingBackend>(calls, true), 8,
                            4);
    });
    require(calls == std::vector<std::string>({"initialize", "destroy"}),
            "failed backend was retained");
    require(!renderer.device() && !renderer.rasterizer(),
            "failed initialization left bound resources");
    calls.clear();
    renderer.initialize(std::make_unique<RecordingBackend>(calls), 8, 4);
    require(!renderer.rasterizer() && !renderer.presentedTexture(),
            "recording backend created Native resources");
    requireRejected([&] {
        renderer.initialize(std::make_unique<RecordingBackend>(calls), 8, 4);
    });
    requireRejected([&] { renderer.readImage(); });
    requireRejected([&] { renderer.syncScene({}); });
    requireRejected([&] { renderer.requestPick({0, 0}); });
    requireRejected([&] { renderer.updateRenderableTransforms(0, {}); });
    requireRejected([&] { renderer.setRenderableExternalBuffer(0, {}); });
    requireRejected([&] { renderer.setRenderableColors(0, {}); });
    requireRejected([&] { renderer.updateRenderableGeometry(0, {}, {}); });
    requireRejected([&] { renderer.updateRenderableSkinningMatrices(0, {}); });
    requireRejected([&] { renderer.setRenderableTexture(0, nullptr, 0); });
    glm::mat4 transform(1);
    requireRejected(
        [&] { renderer.getRenderableInstanceTransform(0, 0, transform); });
    requireRejected(
        [&] { renderer.setRenderableInstanceTransform(0, 0, transform); });
    Camera camera;
    camera.init({0, 0, 4}, {0, 0, 0}, UpAxis::Y);
    camera.updateProjMatrix(8, 4);
    const auto view = camera.getViewMatrix();
    const auto projection = camera.getProjMatrix();
    calls.clear();
    renderer.resizeFrame(16, 8);
    renderer.syncFrame(nullptr, view, projection);
    renderer.prepareFrame(camera, UpAxis::Y);
    renderer.beginFrame();
    requireRejected([&] { renderer.renderScene(view, projection, true); });
    renderer.renderScene(view, projection, false);
    RayPickResult selected;
    selected.hit = true;
    requireRejected([&] { renderer.endFrame(selected, false); });
    requireRejected([&] { renderer.endFrame({}, true); });
    renderer.endFrame({}, false);
    require(calls == std::vector<std::string>({"resize", "sync", "prepare",
                                               "begin", "scene", "end"}),
            "backend frame order changed");
    auto image = renderer.readImage();
    require(image.width == 16 && image.height == 8 &&
                image.pixels.size() == 384,
            "backend capture dimensions changed");
    requireRejected([&] { renderer.addRenderable(nullptr, nullptr); });
    requireRejected([&] { renderer.readCameraPreview(); });
    requireRejected([&] { renderer.native(); });
    requireRejected([&] { renderer.createSceneHookPipeline({}); });
    requireRejected(
        [&] { renderer.addRenderHook(RenderHookPhase::AfterOpaque, {}); });
    requireRejected([&] { renderer.setRenderableTexture(0, nullptr, 0); });
    requireRejected(
        [&] { renderer.mapRenderableCudaTransformBuffers({}, 0, 0, 0); });
    requireRejected(
        [&] { renderer.renderSceneToFramebuffer(camera, nullptr, 8, 4); });
    requireRejected([&] { renderer.readImageResized(0, 4); });
    renderer.release();
    require(calls.back() == "destroy" && image.pixels.front() == 37,
            "release invalidated an owned image");
    const auto count = calls.size();
    renderer.release();
    require(calls.size() == count, "backend was released twice");
    requireRejected([&] { renderer.readImage(); });
}

void checkSceneData() {
    std::vector<std::string> calls;
    auto backend = std::make_unique<RecordingBackend>(calls, false, true);
    auto* recording = backend.get();
    Renderer renderer;
    renderer.initialize(std::move(backend), 8, 4);
    Scene::RenderSceneFrame frame;
    Scene::RenderObjectSource object;
    object.id = 42;
    object.mesh.resource = 7;
    object.mesh.mesh = std::make_shared<Scene::MeshData>();
    object.world[3] = {1, 2, 3, 1};
    frame.objects.push_back(object);
    renderer.syncScene(frame);
    require(recording->snapshot.objects.front().id == 42 &&
                recording->snapshot.objects.front().mesh.mesh ==
                    object.mesh.mesh &&
                recording->snapshot.objects.front().world[3] == object.world[3],
            "scene submission changed identity, source ownership, or world "
            "coordinates");
    frame.objects.front().visible = false;
    renderer.syncScene(frame);
    require(!recording->snapshot.objects.front().visible,
            "scene update lost visibility");
    frame.objects.clear();
    renderer.syncScene(frame);
    require(recording->snapshot.objects.empty(),
            "scene deletion was not forwarded");
    renderer.release();
}

void checkPicking(bool batches) {
    std::vector<std::string> calls;
    ProbeApp app(calls);
    auto backend = std::make_unique<RecordingBackend>(calls, false, !batches,
                                                      batches, true);
    auto* recording = backend.get();
    app.initialize(64, 32, true, UpAxis::Y, Backend::BackendType::OpenGL,
                   Scene::BackendType::Native, true, std::move(backend));
    auto& renderer = app.getRenderer();
    auto& registry = app.getSceneRenderSystem();
    auto& resources = app.getSceneResourceManager();
    auto* scene = app.getScene();
    PBRMaterial material;
    resources.registerMaterial("surface", &material);
    auto mesh =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1));
    const auto meshId = resources.registerMesh("cube", mesh);
    auto* first = scene->definePrim("/first", Scene::PrimType::Mesh);
    auto* second = scene->definePrim("/second", Scene::PrimType::Mesh);
    for (auto* prim : {first, second}) {
        prim->setMeshData(mesh);
        prim->getMeshComponent()->setResourceHandle(meshId);
    }
    auto firstComponent = registry.addRenderable(*first, &material);
    auto secondComponent = registry.addRenderable(*second, &material);
    const auto request = [&] {
        app._mousePickRequested = true;
        app.drawFrame();
        return recording->pickRequest;
    };
    auto old = request();
    require(renderer.hasPendingPick() && app.picks.empty(),
            "request forced synchronous completion");
    require(old.width == 64 && old.height == 32 && old.frame != 0,
            "request lost its view/frame");
    const auto latest = request();
    RenderPickHit hit;
    hit.hit = true;
    hit.instanceIndex = 1;
    if (batches) {
        hit.handle = registry.handle(*secondComponent);
        hit.sceneInstanceId =
            recording->batches.at(hit.handle).instances.ids[1];
    } else {
        for (const auto& object : recording->snapshot.objects) {
            if (object.path == "/second")
                hit.sceneInstanceId = object.id;
        }
    }
    recording->pickResponse = RenderPickResult{old.id, old.frame, hit};
    app.drawFrame();
    require(app.picks.empty() && renderer.hasPendingPick(),
            "superseded response changed selection");

    // Removing a different row must preserve the requested object's identity.
    registry.unregister(*firstComponent);
    recording->pickResponse = RenderPickResult{latest.id, latest.frame, hit};
    app.drawFrame();
    require(app.picks.size() == 1 && app.getSelection().prim == second &&
                (!batches || app.getSelection().instanceIndex == 0),
            "delayed response lost identity after batch compaction");
    require(recording->selection.sceneInstanceId == hit.sceneInstanceId,
            "selection did not cross the pointer-free backend boundary");
    require(registry.buildPrimSelection(*second).sceneInstanceId ==
                hit.sceneInstanceId,
            "programmatic selection used a different identity");

    old = request();
    recording->pickResponse = RenderPickResult{old.id, old.frame, {}};
    app.drawFrame();
    require(app.picks.size() == 2 && !app.getSelection().hit,
            "empty-space click did not clear selection");

    old = request();
    registry.unregister(*secondComponent);
    secondComponent = registry.addRenderable(*second, &material);
    recording->pickResponse = RenderPickResult{old.id, old.frame, hit};
    app.drawFrame();
    require(app.picks.size() == 2 && !app.getSelection().hit,
            "re-registered component accepted its previous identity");

    old = request();
    registry.detachSubtree(*second);
    scene->removePrim("/second");
    recording->pickResponse = RenderPickResult{old.id, old.frame, hit};
    app.drawFrame();
    require(app.picks.size() == 2,
            "deleted object response reached the interaction callback");

    if (batches) {
        auto* external = scene->definePrim("/external", Scene::PrimType::Mesh);
        external->setMeshData(mesh);
        auto component = registry.addRenderable(
            *external, &material, TransformSource::ExternalBuffer);
        registry.updateInstances(*component, {glm::mat4(1)});
        old = request();
        RenderPickHit row;
        row.hit = true;
        row.handle = registry.handle(*component);
        row.instanceIndex = 0;
        row.transformSource = TransformSource::ExternalBuffer;
        recording->pickResponse = RenderPickResult{old.id, old.frame, row};
        registry.updateInstances(*component, {});
        app.drawFrame();
        require(app.picks.size() == 2,
                "changed external rows accepted a delayed response");
    }

    renderer.requestPick({0, 0});
    renderer.resizeFrame(32, 16);
    require(!renderer.hasPendingPick() && !renderer.pollPick(),
            "resize retained a pending pick");
    renderer.requestPick({0, 0});
    renderer.syncFrame(scene, glm::mat4(2.0f), glm::mat4(1.0f));
    require(!renderer.hasPendingPick(), "changed view retained a pending pick");
    renderer.requestPick({0, 0});
    app.clearSelection();
    require(!renderer.hasPendingPick(), "explicit deselection retained a pick");
    app.close();
    require(!renderer.hasPendingPick(), "shutdown retained a pending pick");
    require(std::count(calls.begin(), calls.end(), "fixed") == 0 &&
                std::count(calls.begin(), calls.end(), "update") == 0,
            "picking advanced animation or physics");
}

void checkApp() {
    std::vector<std::string> calls;
    ProbeApp app(calls);
    app.initialize(64, 32, true, UpAxis::Y, Backend::BackendType::OpenGL,
                   Scene::BackendType::Native, true,
                   std::make_unique<RecordingBackend>(calls));
    require(!app.getWindow() && !app.getRenderer().device() &&
                !app.getRasterizer(),
            "custom App still owns Native window/device resources");
    requireRejected([&] { app.logDebugPoints("/points", {}, {}, 1); });
    requireRejected([&] { app.setSkybox(std::string("sky.png")); });
    requireRejected([&] { app.raycast({{0, 0, 1}, {0, 0, -1}}); });
    ColorMaterial material;
    bool rejectedNative = false;
    try {
        Bridge::SkinVisualBridge::fromFBX(&app, &material, "unused.fbx");
    } catch (const std::runtime_error& error) {
        rejectedNative =
            std::string(error.what()).find("Native") != std::string::npos;
    }
    require(rejectedNative, "legacy FBX GPU importer bypassed Native access");
    app.getRenderer().setProfilerEnabled(true);
    calls.clear();
    app.drawFrame();
    require(calls ==
                std::vector<std::string>({"pre", "sync", "prepare", "begin",
                                          "scene", "custom", "end", "post"}),
            "App bypassed the backend or reordered callbacks");
    auto image = app.readImage();
    require(image.width == 64 && image.height == 32 &&
                image.pixels.size() == 6144,
            "App capture depended on a Native texture");
    require(!app.getPresentedTexture(),
            "recording output became a Native texture");
    const auto profile = app.getRenderer().profiler().latestFrameProfile();
    require(profile && !profile->backend &&
                profile->metadata.at("window_visible") == "false",
            "external CPU profiling reported a Native RHI/window");
    require(profile->metadata.count("tone_map_mode") == 0 &&
                profile->metadata.count("culling") == 0,
            "external profile reported unused Native settings");
    app.setFixedUpdateHz(60);
    app.setSimulationPaused(true);
    app.requestSimulationStep();
    calls.clear();
    app.updateAndDraw();
    require(calls == std::vector<std::string>({"update", "fixed", "pre", "sync",
                                               "prepare", "begin", "scene",
                                               "custom", "end", "post"}),
            "windowless App changed update/callback ordering");
    app.processEvents();
    app.requestClose();
    require(app.shouldClose(), "windowless close request was ignored");
    const auto beforeClose = calls;
    app.drawFrame();
    require(calls == beforeClose, "close request still executed a frame");
    app.close();
    require(calls.back() == "destroy" && image.pixels.front() == 37,
            "App close lost backend ownership");
}

void checkBatchRegistration() {
    std::vector<std::string> calls;
    Scene::NativeScene scene;
    Scene::SceneResourceManager resources(&scene);
    Renderer renderer;
    auto backend =
        std::make_unique<RecordingBackend>(calls, false, false, true);
    auto* recording = backend.get();
    renderer.initialize(std::move(backend), 8, 4);
    renderer.bindSceneResources(resources);
    Scene::SceneRenderSystem registry;
    registry.bind(&renderer);
    auto mesh =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1));
    mesh->colors.assign(mesh->vertices.size(), glm::vec3(0.2f, 0.6f, 0.8f));
    PhongMaterial material;
    PBRMaterial replacement;
    const auto materialId = resources.registerMaterial("surface", &material);
    resources.registerMaterial("replacement", &replacement);
    const auto textureId =
        resources.registerTextureSource("source", {"recording.png", true});
    material.textureResources["diffuse_map"] = textureId;
    auto* parent = scene.definePrim("/group", Scene::PrimType::Xform);
    parent->setLocalTranslation({2, 0, 0});
    auto* first = scene.definePrim("/group/first", Scene::PrimType::Mesh);
    auto* second = scene.definePrim("/group/second", Scene::PrimType::Mesh);
    first->setMeshData(mesh);
    second->setMeshData(mesh);
    first->setLocalTranslation({0, 1, 0});
    second->setLocalTranslation({0, 2, 0});
    auto firstComponent = registry.addRenderable(*first, &material);
    auto secondComponent = registry.addRenderable(*second, &material);
    const auto handle = registry.handle(*firstComponent);
    require(recording->batches.size() == 1 &&
                registry.sharesBatch(*firstComponent, *secondComponent),
            "shared mesh/material did not reuse the common backend batch");
    auto sync = [&] {
        renderer.syncFrame(nullptr, glm::mat4(1), glm::mat4(1));
    };
    sync();
    auto& batch = recording->batches.at(handle);
    require(
        batch.mesh.mesh == mesh &&
            batch.material->values.shadingModel ==
                MaterialShadingModel::Phong &&
            batch.instances.transforms.size() == 2 &&
            batch.instances.transforms[0][3] == glm::vec4(2, 1, 0, 1) &&
            batch.instances.transforms[1][3] == glm::vec4(2, 2, 0, 1),
        "batch registration changed geometry, material, or world transforms");
    const auto id = batch.instances.ids[0];
    require(batch.material->resource == materialId &&
                batch.material->textures.at("diffuse_map").first == textureId &&
                batch.material->textures.at("diffuse_map").second->srgb,
            "material source lost catalog identity or texture metadata");
    material.diffuse = {0.2f, 0.4f, 0.7f};
    material.setUnlit(true);
    sync();
    require(batch.material->values.baseColor ==
                    glm::vec4(material.diffuse, 1) &&
                batch.material->values.unlit,
            "legacy material updates were not converted for the backend");
    auto positions = mesh->vertices;
    positions.front().x += 2.0f;
    registry.updateGeometry(*firstComponent, positions);
    require(recording->lastPositions == &positions &&
                mesh->vertices.front() != positions.front(),
            "geometry updates copied input or modified the source mesh");
    requireRejected([&] { registry.setTexture(*firstComponent, nullptr, 0); });
    require(id != 0 && id != batch.instances.ids[1],
            "instance identities are not distinct");
    registry.setDoubleSided(*firstComponent, true);
    registry.setCastsShadow(*firstComponent, false);
    registry.setAlphaMode(*firstComponent, AlphaMode::Blend, 0.3f);
    require(batch.state.doubleSided && !batch.state.castsShadow &&
                batch.state.alphaMode == AlphaMode::Blend &&
                batch.state.alphaCutoff == 0.3f,
            "common batch state lost surface flags");
    first->setDisplayColorAlpha({0.1f, 0.3f, 0.9f, 0.4f});
    parent->setLocalTranslation({4, 0, 0});
    second->setVisible(false);
    sync();
    require(recording->batches.size() == 1 &&
                batch.instances.ids ==
                    std::vector<Scene::RenderInstanceId>{id} &&
                batch.instances.transforms[0][3] == glm::vec4(4, 1, 0, 1) &&
                batch.instances.colors[0] == glm::vec4(0.1f, 0.3f, 0.9f, 0.4f),
            "state updates rebuilt a batch or lost visibility/color/identity");
    registry.setMaterial(*firstComponent, &replacement);
    const auto replacementHandle = registry.handle(*firstComponent);
    sync();
    require(
        replacementHandle != handle && batch.instances.ids.empty() &&
            recording->batches.at(replacementHandle)
                    .material->values.shadingModel == MaterialShadingModel::PBR,
        "material replacement did not transfer the instance between batches");
    auto* external = scene.definePrim("/group/external", Scene::PrimType::Mesh);
    external->setMeshData(mesh);
    auto externalComponent = registry.addRenderable(
        *external, &material, TransformSource::ExternalBuffer);
    const auto externalHandle = registry.handle(*externalComponent);
    require(externalHandle != handle,
            "external and scene transforms shared a batch");
    std::vector<glm::mat4> transforms(2, glm::mat4(1));
    transforms[1][3].x = 9;
    registry.updateInstances(*externalComponent, transforms);
    external->setLocalTranslation({100, 0, 0});
    sync();
    auto& externalBatch = recording->batches.at(externalHandle);
    require(externalBatch.instances.transforms == transforms &&
                externalBatch.source->ids.size() == 1,
            "owner scene transform overwrote external poses");
    external->setVisible(false);
    sync();
    require(externalBatch.source->ids.empty() &&
                externalBatch.instances.transforms == transforms,
            "owner visibility overwrote external data");
    Scene::SkinnedMeshData skin;
    skin.mesh = *mesh;
    skin.boneIndices.assign(mesh->vertices.size(), glm::ivec4(0));
    skin.boneWeights.assign(mesh->vertices.size(), glm::vec4(1, 0, 0, 0));
    auto* skinned = scene.definePrim("/group/skinned", Scene::PrimType::Mesh);
    skinned->setMeshData(std::make_shared<Scene::MeshData>(skin.mesh));
    auto skinComponent =
        registry.addSkinnedRenderable(*skinned, &material, skin);
    sync();
    const auto& skinBatch =
        recording->batches.at(registry.handle(*skinComponent));
    require(skinBatch.boneWeights == skin.boneWeights &&
                skinBatch.instances.ids.size() == 1,
            "skinned registration lost weights or instance identity");
    std::vector<glm::mat4> boneMatrices(1, glm::mat4(1));
    boneMatrices[0][3].y = 0.5f;
    registry.updateSkinning(*skinComponent, boneMatrices);
    require(skinBatch.boneMatrices == boneMatrices &&
                skinned->resolveMeshData()->vertices == skin.mesh.vertices,
            "skinning update bypassed the backend or changed source geometry");
    registry.detachSubtree(*parent);
    scene.removePrim("/group");
    sync();
    for (const auto& [key, removed] : recording->batches)
        require(removed.source->ids.empty(),
                "deleted Prim retained drawable data");
    renderer.release();
}

void checkAppScene(bool native) {
    std::vector<std::string> calls;
    ProbeApp app(calls);
    auto backend = std::make_unique<RecordingBackend>(calls, false, true);
    auto* recording = native ? nullptr : backend.get();
    app.initialize(64, 32, true, UpAxis::Y, Backend::BackendType::OpenGL,
                   Scene::BackendType::Native, true,
                   native ? nullptr : std::move(backend));
    auto& registry = app.getSceneRenderSystem();
    auto& resources = app.getSceneResourceManager();
    auto* scene = app.getScene();
    PBRMaterial material;
    resources.registerMaterial("surface", &material);
    auto mesh =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1));
    const auto meshId = resources.registerMesh("cube", mesh);
    auto* parent = scene->definePrim("/group", Scene::PrimType::Xform);
    parent->setLocalTranslation({1, 0, 0});
    auto* prim = scene->definePrim("/group/cube", Scene::PrimType::Mesh);
    prim->setMeshData(mesh);
    prim->getMeshComponent()->setResourceHandle(meshId);
    prim->setLocalTranslation({0, 2, 0});
    auto component = registry.addRenderable(*prim, &material);
    require(component && registry.isRegistered(*component),
            "scene component registration failed");
    calls.clear();
    app.drawFrame();
    Scene::RenderInstanceId id = 0;
    if (recording) {
        require(calls == std::vector<std::string>({"pre", "scene_data", "sync",
                                                   "prepare", "begin", "scene",
                                                   "custom", "end", "post"}),
                "App did not submit scene data before frame execution");
        require(recording->snapshot.objects.size() == 1,
                "App did not submit its registered object");
        const auto& object = recording->snapshot.objects.front();
        id = object.id;
        require(object.mesh.mesh == mesh && object.mesh.resource == meshId &&
                    object.world[3] == glm::vec4(1, 2, 0, 1),
                "App changed source identity or parent-relative transforms");
        requireRejected([&] { registry.handle(*component); });
        requireRejected([&] { registry.sharesBatch(*component, *component); });
        requireRejected(
            [&] { registry.updateGeometry(*component, mesh->vertices); });
        requireRejected([&] { registry.setTexture(*component, nullptr, 0); });
        requireRejected([&] { app.addRenderable(&material, prim); });
        requireRejected([&] { app.getLight(); });
        requireRejected([&] { app.getRenderer().pointLights(); });
        requireRejected([&] { app.getRenderer().spotLights(); });
        auto* unsupported =
            scene->definePrim("/external", Scene::PrimType::Mesh);
        unsupported->setMeshData(mesh);
        requireRejected([&] {
            registry.addRenderable(*unsupported, &material,
                                   TransformSource::ExternalBuffer);
        });
        require(!unsupported->getRenderComponent(),
                "failed registration left a component");
        prim->getMeshComponent()->setResourceHandle(
            Scene::InvalidResourceHandle);
        requireRejected([&] { app.drawFrame(); });
        require(recording->snapshot.objects.front().id == id,
                "invalid resource changed the last accepted backend scene");
        prim->getMeshComponent()->setResourceHandle(meshId);
        app.drawFrame();
        // An unregistered unsupported component must not enter the backend
        // snapshot.
        unsupported->addRenderComponent()->setTransformSource(
            TransformSource::ExternalBuffer);
    } else {
        require(registry.handle(*component) != InvalidHandle,
                "Native registration lost its compatibility handle");
    }

    parent->setLocalTranslation({3, 0, 0});
    parent->setActive(false);
    parent->setVisible(true);
    registry.setDoubleSided(*component, true);
    material.baseColor = {0.2f, 0.7f, 0.4f, 1};
    app.drawFrame();
    if (recording) {
        const auto& object = recording->snapshot.objects.front();
        require(object.id == id && object.world[3] == glm::vec4(3, 2, 0, 1) &&
                    !object.visible && parent->isVisible() &&
                    object.doubleSided &&
                    object.material.baseColor == material.baseColor,
                "scene update changed identity or lost authored state");
    }
    parent->setActive(true);
    PBRMaterial replacement;
    const auto materialId =
        resources.registerMaterial("replacement", &replacement);
    registry.setMaterial(*component, &replacement);
    app.drawFrame();
    if (recording)
        require(recording->snapshot.objects.front().materialResource ==
                    materialId,
                "material replacement was not submitted");

    registry.unregister(*component);
    app.drawFrame();
    if (recording)
        require(recording->snapshot.objects.empty(),
                "unregistered component remained in the snapshot");
    component = registry.addRenderable(*prim, &replacement);
    app.drawFrame();
    if (recording)
        require(recording->snapshot.objects.front().id != id,
                "removed registration reused an expired instance identity");
    app.removePrim("/group");
    app.drawFrame();
    require(registry.registrationCount() == 0,
            "subtree removal retained registrations");
    if (recording)
        require(recording->snapshot.objects.empty(),
                "subtree removal was not submitted");
    app.close();
}
void checkSceneLights() {
    std::vector<std::string> calls;
    auto backend =
        std::make_unique<RecordingBackend>(calls, false, false, true);
    auto* recording = backend.get();
    Renderer renderer;
    renderer.initialize(std::move(backend), 8, 4);
    Scene::NativeScene scene;
    auto* rig = scene.definePrim("/rig", Scene::PrimType::Xform);
    rig->setLocalTranslation({2, 0, 0});
    auto* prim = scene.definePrim("/rig/light", Scene::PrimType::Light);
    Scene::PointLight light;
    light.position = {1, 0, 0};
    prim->setPointLight(light);
    auto sync = [&] { renderer.syncFrame(&scene, glm::mat4(1), glm::mat4(1)); };
    sync();
    require(recording->lighting.lights.size() == 1,
            "batch backend did not receive a light outside /lights");
    const auto& source = recording->lighting.lights.front();
    require(std::get<Scene::PointLight>(source.light).position.x == 1 &&
                source.world[3].x == 3,
            "light input lost local values or evaluated world transform");
    rig->setVisible(false);
    sync();
    require(recording->lighting.authored && recording->lighting.lights.empty(),
            "hidden scene lights lost ownership or remained visible");
    scene.removePrim("/rig");
    sync();
    require(!recording->lighting.authored && recording->lighting.lights.empty(),
            "last light removal retained stale lighting");
    scene.definePrim("/rig/light", Scene::PrimType::Light)->setRectLight({});
    sync();
    require(recording->lighting.lights.size() == 1 &&
                std::holds_alternative<Scene::RectLight>(
                    recording->lighting.lights.front().light),
            "recreated light reused stale membership");
}
} // namespace

int main(int argc, char** argv) {
    checkSceneLights();
    checkRenderer();
    checkSceneData();
    checkBatchRegistration();
    checkPicking(false);
    checkPicking(true);
    checkApp();
    checkAppScene(false);
    if (argc > 1 && std::string(argv[1]) == "--app") {
        checkAppScene(true);
    }
    std::cout << "PASS: backend initialization, frame order, capabilities, "
                 "capture, release, delayed picking and identity\n";
}

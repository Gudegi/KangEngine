// Checks Native frames, previews, resizing, and teardown without App or UI.
#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"
#include "engine/graphics/renderer/renderer.hpp"
#include "engine/graphics/renderer/render_backend/native/native_render_system.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/native/native_scene.hpp"
#include "engine/scene/scene_resource_manager.hpp"
#include "engine/scene/render_scene_data_collector.hpp"
#include "utils/asset_path.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace KE;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void checkLegacyBinding(Backend::GraphicsDevice& device) {
    Scene::NativeScene scene;
    auto* prim = scene.definePrim("/cube", Scene::PrimType::Mesh);
    prim->setMeshData(
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1)));
    prim->setDisplayColorAlpha({0, 1, 0, 1});
    ColorMaterial material(ColorStyle::Unlit);
    Rasterizer rasterizer(&device);
    Renderer renderer;
    renderer.bind(&device, &rasterizer, nullptr, nullptr);
    renderer.setViewportSize(96, 64);
    renderer.addRenderable(&material, prim);
    Camera camera;
    camera.init({0, 0, 4}, {0, 0, 0}, UpAxis::Y);
    camera.updateProjMatrix(96, 64);
    auto target = device.createFramebuffer({96, 64, false, false, 0});
    int hooks = 0;
    renderer.addRenderHook(RenderHookPhase::AfterOpaque,
                           [&](const RenderHookContext&) { ++hooks; });
    renderer.renderSceneToFramebuffer(camera, target.get(), 96, 64);
    const auto pixels = target->readColorPixels(true);
    const size_t center = (32 * 96 + 48) * 3;
    require(pixels[center + 1] > pixels[center] + 100 && hooks == 1,
            "legacy binding lost offscreen output or render hooks");
    renderer.release();
    require(rasterizer.raycast({{0, 0, 4}, {0, 0, -1}}).hit,
            "facade release destroyed a borrowed Rasterizer");
}

void checkMaterialSources(Backend::GraphicsDevice& device) {
    for (auto model : {MaterialShadingModel::Color, MaterialShadingModel::Phong,
                       MaterialShadingModel::PBR}) {
        Scene::RenderMaterialSource material;
        material.values.shadingModel = model;
        material.values.unlit = true;
        material.values.baseColor = {0, 1, 0, 1};
        Scene::RenderMeshData mesh{
            std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1)),
            1};
        Scene::RenderInstanceData instances;
        instances.transforms = {glm::mat4(1)};
        instances.colors = {model == MaterialShadingModel::Color
                                ? glm::vec4(0, 1, 0, 1)
                                : glm::vec4(1)};
        instances.ids = {1};
        instances.pickable = {0};
        Renderer renderer;
        auto backend = std::make_unique<NativeRenderSystem>(device, renderer);
        auto* native = backend.get();
        renderer.initialize(std::move(backend), 128, 96);
        renderer.rasterizer()->setShadowDistance(0);
        const auto handle =
            native->registerSceneBatch(mesh, material, instances);
        require(handle != InvalidHandle,
                "source-only material failed registration");
        Camera camera;
        camera.init({0, 0, 4}, {0, 0, 0}, UpAxis::Y);
        camera.updateProjMatrix(128, 96);
        auto draw = [&] {
            renderer.syncFrame(nullptr, camera.getViewMatrix(),
                               camera.getProjMatrix());
            renderer.prepareFrame(camera, UpAxis::Y);
            renderer.beginFrame();
            renderer.renderScene(camera.getViewMatrix(), camera.getProjMatrix(),
                                 false);
            renderer.endFrame({}, false);
            return renderer.readImage();
        };
        const auto green = draw();
        const size_t center = (48 * 128 + 64) * 3;
        require(green.pixels[center + 1] > green.pixels[center + 2] + 100,
                "CPU material values did not reach Native output");
        material.values.baseColor = {0, 0, 1, 1};
        if (model == MaterialShadingModel::Color)
            instances.colors.front() = material.values.baseColor;
        const auto blue = draw();
        require(blue.pixels[center + 2] > blue.pixels[center + 1] + 100,
                "borrowed material updates were not applied");
        if (model != MaterialShadingModel::Color) {
            material.values.baseColor = {1, 0, 0, 1};
            renderer.renderCameraPreview(camera.getViewMatrix(),
                                         camera.getProjMatrix(), 128, 96);
            const auto preview = renderer.readCameraPreview();
            require(preview.pixels[center] > preview.pixels[center + 2] + 100,
                    "camera preview ignored source material updates");
            material.values.baseColor = glm::vec4(1);
            const auto plain = draw();
            auto texture = std::make_shared<Scene::TextureSourceResource>();
            texture->path = getAssetPath("textures/awesomeface.png");
            texture->srgb = true;
            const std::string slot = model == MaterialShadingModel::Phong
                                         ? "diffuse_map"
                                         : "base_color_texture";
            material.textures[slot] = {2, texture};
            const auto textured = draw();
            require(textured.pixels != plain.pixels,
                    "source file texture was not realized");
            device.profileContext()->beginFrame(1, 1,
                                                Backend::BackendType::OpenGL);
            draw();
            require(device.profileContext()
                            ->endFrame()
                            ->counters.textureUploadBytes == 0,
                    "unchanged source texture was reuploaded");
            material.textures.clear();
            require(draw().pixels == plain.pixels,
                    "removed source texture remained bound");
        }
        renderer.release();
    }
}

void checkDynamicBatches(Backend::GraphicsDevice& device) {
    for (bool skinned : {false, true}) {
        for (auto source :
             {TransformSource::SceneGraph, TransformSource::ExternalBuffer}) {
            Scene::NativeScene scene;
            ColorMaterial material(ColorStyle::Unlit);
            auto mesh = std::make_shared<Scene::MeshData>(
                Scene::Prim::createCubeData(1));
            const auto sourceVertices = mesh->vertices;
            auto* prim = scene.definePrim("/dynamic", Scene::PrimType::Mesh);
            prim->setMeshData(mesh);
            Scene::SkinnedMeshData skin;
            skin.mesh = *mesh;
            skin.boneIndices.assign(mesh->vertices.size(), glm::ivec4(0));
            skin.boneWeights.assign(mesh->vertices.size(),
                                    glm::vec4(1, 0, 0, 0));
            Renderer renderer;
            renderer.initialize(device, 128, 96);
            renderer.rasterizer()->setShadowDistance(0);
            const auto handle =
                skinned ? renderer.addSkinnedRenderable(&material, prim, skin,
                                                        source)
                        : renderer.addRenderable(&material, prim, source);
            std::vector<glm::mat4> transforms(1, glm::mat4(1));
            if (source == TransformSource::ExternalBuffer)
                renderer.updateRenderableTransforms(handle, transforms);
            if (skinned)
                renderer.updateRenderableSkinningMatrices(handle, transforms);
            Camera camera;
            camera.init({0, 0, 4}, {0, 0, 0}, UpAxis::Y);
            camera.updateProjMatrix(128, 96);
            auto pick = [&] {
                renderer.syncFrame(&scene, camera.getViewMatrix(),
                                   camera.getProjMatrix());
                return renderer.rasterizer()->pickScreen(
                    {0, 0}, camera.getViewMatrix(), camera.getProjMatrix(), 128,
                    96);
            };
            auto hit = pick();
            require(
                hit.hit && hit.handle == handle &&
                    hit.transformSource == source &&
                    hit.prim == (source == TransformSource::SceneGraph
                                     ? prim
                                     : nullptr),
                "dynamic batch lost picking identity or transform ownership");
            prim->setVisible(false);
            require(!pick().hit, "hidden owner left its dynamic batch visible");
            prim->setVisible(true);
            prim->setActive(false);
            require(!pick().hit && prim->isVisible(),
                    "active changed visibility or left a drawable batch");
            prim->setActive(true);
            require(pick().hit, "reactivated dynamic batch stayed hidden");
            prim->getSelectionComponent()->setPickable(false);
            require(!pick().hit, "dynamic owner pickability was ignored");
            prim->getSelectionComponent()->setPickable(true);
            if (source == TransformSource::ExternalBuffer) {
                prim->setLocalTranslation({10, 0, 0});
                require(pick().hit, "owner transform overwrote external poses");
            }
            if (skinned) {
                transforms[0][3].x = 10;
                renderer.updateRenderableSkinningMatrices(handle, transforms);
            } else {
                auto positions = sourceVertices;
                for (auto& point : positions)
                    point.x += 10;
                renderer.updateRenderableGeometry(handle, positions, {});
            }
            require(!pick().hit && mesh->vertices == sourceVertices,
                    "deformation failed or changed the source mesh");
            renderer.removePrim(handle, prim);
            scene.removePrim("/dynamic");
            require(!pick().hit, "deleted dynamic owner remained queryable");
            renderer.release();
        }
    }
}

void checkSceneData(Backend::GraphicsDevice& device) {
    Renderer renderer;
    renderer.initialize(device, 128, 96);
    renderer.rasterizer()->setShadowDistance(0);
    renderer.settings().background.backgroundColor = {0, 0, 0, 1};
    Camera camera;
    camera.init({0, 0, 4}, {0, 0, 0}, UpAxis::Y);
    camera.updateProjMatrix(128, 96);
    Scene::RenderSceneFrame scene;
    Scene::RenderObjectSource cube;
    cube.id = 1;
    cube.mesh = {
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1)), 1};
    cube.materialResource = 2;
    cube.material.shadingModel = MaterialShadingModel::PBR;
    cube.material.baseColor = {0.1f, 0.8f, 0.2f, 1};
    scene.objects.push_back(cube);
    Scene::DirectionalLight light;
    light.direction = {0, 0, -1};
    light.intensity = 3;
    light.ambient = glm::vec3(0);
    scene.lights.push_back(light);

    auto submit = [&] {
        auto profile = device.profileContext();
        profile->beginFrame(1, 1, Backend::BackendType::OpenGL);
        renderer.syncScene(scene);
        return profile->endFrame()->counters;
    };
    auto draw = [&] {
        const auto view = camera.getViewMatrix();
        const auto projection = camera.getProjMatrix();
        renderer.syncFrame(nullptr, view, projection);
        renderer.prepareFrame(camera, UpAxis::Y);
        renderer.beginFrame();
        renderer.renderScene(view, projection, false);
        renderer.endFrame({}, false);
        return renderer.readImage();
    };

    require(submit().bufferAllocations > 0,
            "new scene did not upload geometry");
    const auto initial = draw();
    const size_t center = (48 * 128 + 64) * 3;
    require(initial.pixels[center + 1] > initial.pixels[center] + 30,
            "source scene was not rendered");
    const auto unchanged = submit();
    require(unchanged.bufferAllocations == 0 &&
                unchanged.bufferUploadBytes == 0 &&
                unchanged.textureUploadBytes == 0,
            "unchanged registration recreated or reuploaded resources");
    require(draw().pixels == initial.pixels, "unchanged scene changed output");

    auto second = cube;
    second.id = 2;
    second.world[3].x = 1.1f;
    scene.objects.push_back(second);
    require(submit().bufferAllocations == 0,
            "shared mesh/material created another GPU batch");
    require(draw().pixels != initial.pixels, "second instance was not drawn");
    scene.objects.pop_back();
    submit();
    require(draw().pixels == initial.pixels,
            "removed instance remained in its batch");

    scene.objects.front().world[3].x = 0.7f;
    require(submit().bufferAllocations == 0, "pose update rebuilt geometry");
    require(draw().pixels != initial.pixels, "world pose was not applied");
    scene.objects.front().visible = false;
    submit();
    const auto hidden = draw();
    scene.objects.clear();
    submit();
    require(draw().pixels == hidden.pixels, "deleted geometry stayed visible");
    scene.objects.push_back(cube);
    require(submit().bufferAllocations > 0,
            "deleted batch retained GPU buffers");
    require(draw().pixels == initial.pixels,
            "recreated scene lost its resources");

    scene.objects.front().material.unlit = true;
    bool rejected = false;
    try {
        renderer.syncScene(scene);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected && draw().pixels == initial.pixels,
            "unsupported data changed the accepted scene");
    renderer.release();
}
} // namespace

int main() {
    Window window;
    window.init(128, 96, true);
    Backend::OpenGLDevice device;
    device.initialize();
    device.setValidationEnabled(true);
    Scene::NativeScene scene;
    Scene::SceneResourceManager resources(&scene);
    ColorMaterial material(ColorStyle::Unlit);
    auto mesh =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1));
    auto* prim = scene.definePrim("/cube", Scene::PrimType::Mesh);
    prim->setMeshData(mesh);
    prim->setDisplayColorAlpha({0.1f, 0.8f, 0.2f, 1});
    prim->setLocalTranslation({0, 0.3f, 0});
    prim->getMeshComponent()->setResourceHandle(
        resources.registerMesh("cube", mesh));
    resources.registerMaterial("cube", &material);
    Camera camera;
    camera.init({0, 0, 4}, {0, 0, 0}, UpAxis::Y);
    Renderer renderer;
    renderer.settings().background.backgroundColor = {0, 0, 0, 1};

    for (int iteration = 0; iteration < 2; ++iteration) {
        {
            renderer.initialize(device, 128, 96);
            require(!renderer.presentedTexture(),
                    "output exists before a frame");
            require(renderer.sampleCount() == 4, "scene MSAA changed");
            renderer.rasterizer()->setShadowDistance(0);
            const auto handle = renderer.addRenderable(&material, prim);
            int hookCalls = 0;
            int width = 128;
            int height = 96;
            renderer.addRenderHook(
                RenderHookPhase::AfterTransparent,
                [&](const RenderHookContext& context) {
                    require(context.width == width && context.height == height,
                            "render hook used stale dimensions");
                    ++hookCalls;
                });

            for (const auto size : {glm::ivec2(128, 96), glm::ivec2(80, 64)}) {
                width = size.x;
                height = size.y;
                renderer.resizeFrame(width, height);
                require(!renderer.presentedTexture(),
                        "resize exposed an unfinished frame");
                camera.updateProjMatrix(width, height);
                const auto view = camera.getViewMatrix();
                const auto projection = camera.getProjMatrix();
                renderer.syncFrame(&scene, view, projection);
                const auto request = renderer.requestPick({0, 0});
                const auto response = renderer.pollPick();
                require(response && response->requestId == request &&
                            response->hit.hit &&
                            renderer.resolvePick(response->hit).prim == prim,
                        "Native screen picking bypassed the common result");
                const auto hit =
                    renderer.rasterizer()->raycast({{0, 0.3f, 4}, {0, 0, -1}});
                require(
                    hit.hit && hit.prim == prim &&
                        hit.transformSource == TransformSource::SceneGraph,
                    "common batch lost its raycast Prim or transform owner");
                renderer.prepareFrame(camera, UpAxis::Y);
                renderer.beginFrame();
                renderer.renderScene(view, projection, false);
                renderer.endFrame({}, false);
                auto* output = renderer.presentedTexture();
                const auto image = renderer.readImage(true);
                const auto& pixels = image.pixels;
                require(image.width == width && image.height == height,
                        "readback dimensions do not match its pixels");
                require(pixels.size() ==
                            static_cast<size_t>(width * height * 3),
                        "frame dimensions did not follow resize");
                const size_t center = (height / 2 * width + width / 2) * 3;
                require(pixels[center + 1] > pixels[center] + 30,
                        "scene draw did not reach the resolved output");
                const auto bottomUp = renderer.readImage(false);
                require(bottomUp.pixels != pixels,
                        "fixture must distinguish image row directions");
                const size_t rowSize = static_cast<size_t>(width) * 3;
                for (int row = 0; row < height; ++row)
                    require(std::equal(pixels.begin() + row * rowSize,
                                       pixels.begin() + (row + 1) * rowSize,
                                       bottomUp.pixels.begin() +
                                           (height - 1 - row) * rowSize),
                            "readback row direction changed");
                const auto reduced = renderer.readImageResized(20, 12, true);
                require(reduced.width == 20 && reduced.height == 12 &&
                            reduced.pixels.size() == 20 * 12 * 3,
                        "resized readback lost dimensions or RGB layout");

                width = 40;
                height = 32;
                auto* preview = renderer.renderCameraPreview(view, projection,
                                                             width, height);
                require(preview->getWidth() == width &&
                            preview->getHeight() == height,
                        "preview dimensions are wrong");
                require(renderer.readCameraPreview(true).pixels.size() ==
                            static_cast<size_t>(width * height * 3),
                        "preview readback dimensions are wrong");
                require(renderer.presentedTexture() == output &&
                            renderer.readImage(true).pixels == pixels,
                        "preview overwrote the main frame");
            }
            require(hookCalls == 4,
                    "main and preview hooks changed order/count");
            auto saved = renderer.readImage(true);
            const auto savedPixels = saved.pixels;
            renderer.removePrim(handle, prim);
            renderer.syncFrame(&scene, camera.getViewMatrix(),
                               camera.getProjMatrix());
            require(
                !renderer.rasterizer()->raycast({{0, 0.3f, 4}, {0, 0, -1}}).hit,
                "removed common batch instance remained queryable");
            renderer.release();
            require(saved.pixels == savedPixels && saved.width == 80 &&
                        saved.height == 64,
                    "readback did not survive renderer release");
            require(!renderer.presentedTexture(),
                    "released frame remained visible");
            bool rejected = false;
            try {
                renderer.readImage(true);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            require(rejected, "readback accepted a released renderer");
        }
        require(!renderer.device() && !renderer.rasterizer() &&
                    !renderer.postProcessor() && !renderer.selectionOutline(),
                "facade retained destroyed Native resources");
    }
    checkSceneData(device);
    checkDynamicBatches(device);
    checkMaterialSources(device);
    checkLegacyBinding(device);
    std::cout << "PASS: Native scene data, frames without App, resize, hooks, "
                 "preview, "
                 "teardown\n";
}

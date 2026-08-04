#include "engine/graphics/material/material.hpp"
#include "engine/scene/native/prim.hpp"
#include "kangEngine.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <dlfcn.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace KE;

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct RenderDocApi100Prefix {
    void* entriesBeforeStartFrameCapture[19];
    void (*startFrameCapture)(void*, void*);
    uint32_t (*isFrameCapturing)();
    uint32_t (*endFrameCapture)(void*, void*);
};

RenderDocApi100Prefix* injectedRenderDocApi() {
    using GetApi = int (*)(int, void**);
    auto getApi = reinterpret_cast<GetApi>(
        dlsym(RTLD_DEFAULT, "RENDERDOC_GetAPI"));
    if (!getApi)
        return nullptr;
    RenderDocApi100Prefix* api = nullptr;
    return getApi(10000, reinterpret_cast<void**>(&api)) == 1 ? api : nullptr;
}

class Phase1SceneSmoke final : public App {
  public:
    void setup() override {
        getCamera().setCameraPos({0.0f, 3.0f, 8.0f});
        getCamera().setTargetPos({0.0f, 0.5f, 0.0f});
        setLight({glm::normalize(glm::vec3(0.4f, 1.0f, 0.25f)),
                  glm::vec3(1.0f), 1.0f, glm::vec3(0.12f)});

        createMaterialTexture();
        createMaterials();
        createStaticScene();
        createSkinnedScene();
        createTexturedVertexColorScene();
        createVisualBridgeScene();
        createExternalInstanceBatch();
        createExperimentalRenderHook();

        logDebugLines("/smoke/debug_lines", {{-3.0f, 0.1f, -1.0f}},
                      {{3.0f, 0.1f, -1.0f}}, {{1.0f, 0.2f, 0.1f, 1.0f}},
                      2.0f, false);
        logDebugPoints("/smoke/debug_points", {{0.0f, 1.8f, 0.0f}},
                       {{0.2f, 0.8f, 1.0f, 1.0f}}, 7.0f, false);
        ScreenTextDesc statusText;
        statusText.text = "RHI Phase 1";
        statusText.position = {12.0f, 12.0f};
        statusText.color = {1.0f, 1.0f, 1.0f, 1.0f};
        statusText.pixelSize = 18.0f;
        setScreenText("/smoke/status", statusText);

        getSelectionOutlineProcessor()->config().enabled = true;
        getSelectionOutlineProcessor()->config().radius = 2.0f;
        getRasterizer()->setFrustumCullingEnabled(true);

        Backend::FramebufferDesc previewDesc;
        previewDesc.width = 96;
        previewDesc.height = 64;
        previewDesc.stencil = true;
        previewDesc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
        previewFramebuffer =
            getGraphicsDevice()->createFramebuffer(previewDesc);
        previewDesc.colorFormat = Backend::FramebufferColorFormat::RGBA8;
        rgba8PreviewFramebuffer =
            getGraphicsDevice()->createFramebuffer(previewDesc);
    }

    void createExperimentalRenderHook() {
        static constexpr glm::vec2 vertices[] = {
            {-0.12f, 0.72f}, {0.12f, 0.72f}, {0.0f, 0.92f}};
        Backend::BufferDesc bufferDesc;
        bufferDesc.size = sizeof(vertices);
        bufferDesc.usage = Backend::BufferUsage::Vertex;
        bufferDesc.label = "phase1_render_hook_vertices";
        renderHookVertexBuffer =
            getGraphicsDevice()->createBuffer(bufferDesc, vertices);

        Backend::VertexBufferLayout vertexLayout;
        vertexLayout.arrayStride = sizeof(glm::vec2);
        vertexLayout.attributes = {
            {Backend::VertexFormat::Float32x2, 0, 0}};
        SceneHookPipelineDesc pipelineDesc;
        pipelineDesc.label = "phase1_render_hook_pipeline";
        pipelineDesc.shader.name = "phase1_render_hook_shader";
        pipelineDesc.shader.stages = {
            {R"(#version 410 core
                layout(location = 0) in vec2 aPosition;
                void main() { gl_Position = vec4(aPosition, 0.0, 1.0); })",
             Backend::ShaderType::Vertex, "main"},
            {R"(#version 410 core
                layout(location = 0) out vec4 outColor;
                void main() { outColor = vec4(1.0, 0.1, 0.8, 1.0); })",
             Backend::ShaderType::Fragment, "main"},
        };
        pipelineDesc.vertexBuffers = {vertexLayout};
        pipelineDesc.depthTest = false;
        renderHookPipeline =
            getRenderer().createSceneHookPipeline(pipelineDesc);

        renderHook = getRenderer().addRenderHook(
            RenderHookPhase::AfterTransparent,
            [this](RenderHookContext& context) {
                ++renderHookRecordCount;
                context.pass.setViewport(
                    0.0f, 0.0f, static_cast<float>(context.width),
                    static_cast<float>(context.height));
                context.pass.setPipeline(renderHookPipeline.get());
                context.pass.setVertexBuffer(0, renderHookVertexBuffer.get());
                context.pass.draw(3);
            });
        require(renderHook != InvalidRenderHook,
                "failed to register experimental render hook");
    }

    void preRender() override {
        // Exercise live material, skin, visibility, and instance-buffer updates
        // without changing resource ownership while commands are recorded.
        const float x = 0.04f * static_cast<float>(frame);
        updateRenderableSkinningMatrices(skinnedPhongHandle,
                                        {glm::translate(glm::mat4(1.0f),
                                                        glm::vec3(x, 0, 0))});
        updateRenderableSkinningMatrices(skinnedPbrHandle, {glm::mat4(1.0f)});
        skinBridge.applyTime(0.01f * static_cast<float>(frame), true);
        skeletalBridge.applyMotion(bvhMotion,
                                   0.01f * static_cast<float>(frame), true);
        visibilityPrim->setVisible(frame != 1);
        getSelectionOutlineProcessor()->config().radius = 1.0f + frame;
        if (frame == 3) {
            renderDocApi = injectedRenderDocApi();
            if (renderDocApi && renderDocApi->startFrameCapture)
                renderDocApi->startFrameCapture(nullptr, nullptr);
        }
    }

    void postRender() override {
        checkError();
        require(getPresentedTexture() != nullptr,
                "production scene did not produce a presented texture");

        if (frame == 0) {
            // Instance indices are assigned by the first MeshInstancer update.
            // Select after that update so the next frame exercises the real
            // skinned alpha-mask selection pass instead of the prim fallback.
            selectPrim(selectedPrim);
            require(hasSelection(), "selected skinned prim was not registered");
            require(getSelection().instanceIndex == 0,
                    "selected prim instance index mismatch");
            // Exercise the editor panel's RHI wireframe state on the next
            // frame, then restore fill before offscreen validation.
            _renderWireframe = true;
            framebufferSizeCallback(getWindow(), 400, 240);
        } else if (frame == 1) {
            _renderWireframe = false;
            framebufferSizeCallback(getWindow(), 257, 193);
        } else if (frame == 2) {
            getRenderer().renderSceneToFramebuffer(
                getCamera(), previewFramebuffer.get(), 96, 64, true);
            getRenderer().renderSceneToFramebuffer(
                getCamera(), rgba8PreviewFramebuffer.get(), 96, 64, true);
            const auto rgba8 = readFramebuffer(
                *rgba8PreviewFramebuffer, "phase1_rgba8_offscreen_view",
                Backend::TextureFormat::RGBA8Unorm);
            double rgba8Energy = 0.0;
            for (float value : rgba8.values) {
                require(std::isfinite(value),
                        "RGBA8 offscreen RHI output contains non-finite pixels");
                rgba8Energy += std::abs(static_cast<double>(value));
            }
            require(!rgba8.values.empty() &&
                        rgba8Energy / rgba8.values.size() > 0.01,
                    "RGBA8 offscreen RHI output is empty");
            previewFramebuffer->resize(81, 53);
            getRenderer().renderSceneToFramebuffer(
                getCamera(), previewFramebuffer.get(), 81, 53, true);
            require(previewFramebuffer->getColorTexture()->getWidth() == 81,
                    "offscreen RHI scene target resize failed");
            framebufferSizeCallback(getWindow(), 320, 180);
        }
        else if (frame == 3 && renderDocApi) {
            require(renderDocApi->endFrameCapture(nullptr, nullptr) == 1,
                    "RenderDoc failed to save the Phase 1 scene capture");
        }
        else if (frame == 4) {
            require(getWidth() == 320 && getHeight() == 180,
                    "repeated framebuffer resize did not settle");
            require(getRasterizer()->getCullingTotalInstances() >= 2,
                    "frustum culling did not inspect the external batch");
            require(getRasterizer()->getCullingCulledInstances() >= 1,
                    "off-frustum external instance was not culled");
            require(hasSelection() && getSelection().prim == selectedPrim,
                    "selection did not survive resize/render cycles");
            require(renderHookRecordCount > 0,
                    "experimental render hook did not record commands");
            require(getRenderer().removeRenderHook(renderHook),
                    "experimental render hook removal failed");
            renderHook = InvalidRenderHook;
            completed = true;
            requestClose();
        }
        ++frame;
    }

    bool completed = false;

  private:
    Scene::Prim* addMaterialPrim(const std::string& path, Material* material,
                                 const glm::vec3& position, bool doubleSided,
                                 AlphaMode alphaMode) {
        auto* prim = getScene()->definePrim(path, Scene::PrimType::Mesh);
        prim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createCubeData(0.8f)));
        prim->setLocalTranslation(position);
        const auto handle = addRenderable(material, prim);
        require(handle != InvalidHandle, "static material registration failed");
        setRenderableDoubleSided(handle, doubleSided);
        setRenderableAlphaMode(handle, alphaMode, 0.35f);
        return prim;
    }

    std::pair<Scene::Prim*, RenderableHandle>
    addSkinnedMaterialPrim(const std::string& path, Material* material,
                           const glm::vec3& position, AlphaMode alphaMode) {
        Scene::SkinnedMeshData skin;
        skin.mesh = Scene::Prim::createCubeData(0.8f);
        skin.boneIndices.assign(skin.mesh.vertices.size(), glm::ivec4(0));
        skin.boneWeights.assign(skin.mesh.vertices.size(),
                                glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        auto* prim = getScene()->definePrim(path, Scene::PrimType::Mesh);
        prim->setMeshData(std::make_shared<Scene::MeshData>(skin.mesh));
        prim->setLocalTranslation(position);
        const auto handle = addSkinnedRenderable(material, prim, skin);
        require(handle != InvalidHandle, "skinned material registration failed");
        setRenderableAlphaMode(handle, alphaMode, 0.35f);
        updateRenderableSkinningMatrices(handle, {glm::mat4(1.0f)});
        return {prim, handle};
    }

    void createMaterialTexture() {
        const std::vector<unsigned char> pixels = {
            255, 255, 255, 255, 64, 180, 255, 255,
            255, 80, 80, 255,    255, 255, 255, 0,
        };
        Backend::TextureResourceDesc desc;
        desc.extent = {2, 2, 1};
        desc.format = Backend::TextureFormat::RGBA8Unorm;
        desc.usage = Backend::TextureUsage::TextureBinding |
                     Backend::TextureUsage::CopyDst;
        desc.label = "phase1_scene_material_texture";
        Backend::TextureInitialData initial{pixels.data(), pixels.size(), 8};
        materialTexture = getGraphicsDevice()->createTexture(desc, &initial);
    }

    void createMaterials() {
        phong = std::make_unique<PhongMaterial>();
        phong->setDiffuse({0.8f, 0.35f, 0.2f})
            ->setDiffuseMap(materialTexture.get())
            ->setSpecularMap(materialTexture.get())
            ->setAlphaMap(materialTexture.get())
            ->setNormalMap(materialTexture.get());

        pbr = std::make_unique<PBRMaterial>();
        pbr->setBaseColor({0.2f, 0.65f, 0.9f, 1.0f})
            ->setMetallic(0.25f)
            ->setRoughness(0.55f)
            ->setBaseColorTexture(materialTexture.get())
            ->setNormalTexture(materialTexture.get())
            ->setMetallicRoughnessTexture(materialTexture.get())
            ->setMetallicTexture(materialTexture.get())
            ->setRoughnessTexture(materialTexture.get())
            ->setAoTexture(materialTexture.get())
            ->setOrmTexture(materialTexture.get())
            ->setEmissiveTexture(materialTexture.get());

        skinnedPhong = std::make_unique<PhongMaterial>();
        skinnedPhong->setDiffuseMap(materialTexture.get())
            ->setAlphaMap(materialTexture.get())
            ->setNormalMap(materialTexture.get());
        skinnedPbr = std::make_unique<PBRMaterial>();
        skinnedPbr->setBaseColorTexture(materialTexture.get())
            ->setNormalTexture(materialTexture.get())
            ->setOrmTexture(materialTexture.get());
    }

    void createStaticScene() {
        MeshPrimDesc ground;
        ground.material = &checkerMaterial;
        ground.path = "/smoke/checkerboard_ground";
        ground.meshData = Scene::Prim::createPlaneData(12.0f, UpAxis::Y);
        ground.doubleSided = true;
        addMeshPrim(std::move(ground));
        addMaterialPrim("/smoke/phong_opaque", phong.get(), {-2.0f, 0.5f, 0.0f},
                        false, AlphaMode::Opaque);
        visibilityPrim = addMaterialPrim("/smoke/pbr_double_sided", pbr.get(),
                                         {0.0f, 0.5f, 0.0f}, true,
                                         AlphaMode::Opaque);
        addMaterialPrim("/smoke/phong_mask", phong.get(), {2.0f, 0.5f, 0.0f},
                        false, AlphaMode::Mask);
        addMaterialPrim("/smoke/pbr_blend", pbr.get(), {0.0f, 0.5f, 1.5f},
                        true, AlphaMode::Blend);
        addMaterialPrim("/smoke/debug_checker", &debugCheckerMaterial,
                        {-2.8f, 0.5f, 1.5f}, true, AlphaMode::Opaque);
    }

    void createSkinnedScene() {
        auto phongResult = addSkinnedMaterialPrim(
            "/smoke/skinned_phong_mask", skinnedPhong.get(),
            {-1.1f, 1.5f, 0.0f}, AlphaMode::Mask);
        selectedPrim = phongResult.first;
        skinnedPhongHandle = phongResult.second;
        auto pbrResult = addSkinnedMaterialPrim(
            "/smoke/skinned_pbr_blend", skinnedPbr.get(),
            {1.1f, 1.5f, 0.0f}, AlphaMode::Blend);
        skinnedPbrHandle = pbrResult.second;
        addSkinnedMaterialPrim("/smoke/skinned_debug_checker",
                               &debugCheckerMaterial, {2.8f, 1.5f, 1.5f},
                               AlphaMode::Opaque);
    }

    void createTexturedVertexColorScene() {
        auto* staticPrim =
            getScene()->definePrim("/smoke/textured_vertex_color",
                                   Scene::PrimType::Mesh);
        staticPrim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createCubeData(0.65f)));
        staticPrim->setLocalTranslation({-2.8f, 1.4f, 0.0f});
        const auto staticHandle = addRenderable(&texturedMaterial, staticPrim);
        require(staticHandle != InvalidHandle,
                "textured vertex-color registration failed");
        setRenderableTexture(staticHandle, materialTexture.get(),
                             TextureRole::BaseColor);
        setRenderableTexture(staticHandle, materialTexture.get(),
                             TextureRole::Normal);
        setRenderableAlphaMode(staticHandle, AlphaMode::Blend, 0.35f);

        Scene::SkinnedMeshData skin;
        skin.mesh = Scene::Prim::createCubeData(0.65f);
        skin.boneIndices.assign(skin.mesh.vertices.size(), glm::ivec4(0));
        skin.boneWeights.assign(skin.mesh.vertices.size(),
                                glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        auto* skinPrim = getScene()->definePrim(
            "/smoke/skinned_textured_vertex_color", Scene::PrimType::Mesh);
        skinPrim->setMeshData(std::make_shared<Scene::MeshData>(skin.mesh));
        skinPrim->setLocalTranslation({2.8f, 1.4f, 0.0f});
        const auto skinHandle =
            addSkinnedRenderable(&texturedMaterial, skinPrim, skin);
        require(skinHandle != InvalidHandle,
                "skinned textured vertex-color registration failed");
        setRenderableTexture(skinHandle, materialTexture.get(),
                             TextureRole::BaseColor);
        setRenderableTexture(skinHandle, materialTexture.get(),
                             TextureRole::Normal);
        setRenderableAlphaMode(skinHandle, AlphaMode::Mask, 0.35f);
        updateRenderableSkinningMatrices(skinHandle, {glm::mat4(1.0f)});
    }

    void createExternalInstanceBatch() {
        auto* prim = getScene()->definePrim("/smoke/external", Scene::PrimType::Mesh);
        prim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createCubeData(0.3f)));
        externalHandle =
            addRenderable(&commonMaterial, prim,
                          TransformSource::ExternalBuffer);
        require(externalHandle != InvalidHandle,
                "external-buffer renderable registration failed");
        const std::vector<glm::mat4> transforms = {
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.3f, -1.5f)),
            glm::translate(glm::mat4(1.0f), glm::vec3(500.0f, 0.0f, 0.0f)),
        };
        updateRenderableTransforms(externalHandle, transforms);
        glm::mat4 first;
        require(getRenderableInstanceTransform(externalHandle, 0, first),
                "external instance transform query failed");
    }

    void createVisualBridgeScene() {
        skinBridge = Bridge::SkinVisualBridge::fromFBX(
            this, &texturedMaterial,
            getAssetPath("external/Capoeira.fbx"), "/smoke/skin_bridge", -1,
            -1.0f, 0.01f, true);
        require(!skinBridge.meshes().empty(),
                "SkinVisualBridge imported no skinned meshes");

        bvhMotion = loadBVHMotion(
            getAssetPath(
                "external/SMPL_AMASS_T_HDM_bk_01-01_01_120_poses.bvh"),
            1.0f, "/smoke/bvh_motion");
        Bridge::SkeletalVisualConfig config;
        config.visible = true;
        config.showJoints = true;
        skeletalBridge = Bridge::SkeletalVisualBridge::define(
            this, &commonMaterial, "/smoke/skeletal_bridge", bvhMotion,
            0.0f, true, config);
        require(skeletalBridge.boneHandle() != InvalidHandle &&
                    skeletalBridge.jointHandle() != InvalidHandle,
                "SkeletalVisualBridge renderables were not registered");

        const auto mjcf = Asset::MJCFLoader::load(getAssetPath(
            "external/retargetted/unitree_h1/unitree_h1.xml"));
        auto articulationAsset =
            Bridge::ArticulationVisualBridgeAsset::fromData(mjcf);
        articulationAsset.defineMeshAssets(
            getScene(), "/smoke/articulation_mesh_assets");
        articulationBridge = articulationAsset.instantiate(
            getScene(), "/smoke/articulation_bridge",
            "/smoke/articulation_mesh_assets");
        require(!articulationBridge.bodyPrims().empty(),
                "ArticulationVisualBridge imported no body prims");
        for (auto* body : articulationBridge.bodyPrims()) {
            const auto handle = addRenderable(
                &commonMaterial, body, TransformSource::ExternalBuffer);
            require(handle != InvalidHandle,
                    "ArticulationVisualBridge external registration failed");
            updateRenderableTransforms(handle, {glm::mat4(1.0f)});
            articulationHandles.push_back(handle);
        }
    }

    Backend::TextureReadback readFramebuffer(Backend::Framebuffer& framebuffer,
                                             const char* label,
                                             Backend::TextureFormat format =
                                                 Backend::TextureFormat::RGBA16Float) {
        Backend::TextureViewDesc viewDesc;
        viewDesc.format = format;
        viewDesc.label = label;
        auto view = getGraphicsDevice()->createTextureView(
            framebuffer.getColorTexture(), viewDesc);
        return getGraphicsDevice()->readTexture(view.get());
    }

    int frame = 0;
    Scene::Prim* selectedPrim = nullptr;
    Scene::Prim* visibilityPrim = nullptr;
    RenderableHandle skinnedPhongHandle = InvalidHandle;
    RenderableHandle skinnedPbrHandle = InvalidHandle;
    RenderableHandle externalHandle = InvalidHandle;
    RenderDocApi100Prefix* renderDocApi = nullptr;
    Bridge::SkinVisualBridge skinBridge;
    Bridge::SkeletalVisualBridge skeletalBridge;
    Bridge::ArticulationVisualBridge articulationBridge;
    std::vector<RenderableHandle> articulationHandles;
    Animation::SkeletonMotion bvhMotion;

    VertexColorMaterial commonMaterial;
    VertexColorMaterial texturedMaterial{VertexColorStyle::Textured};
    VertexColorMaterial checkerMaterial{VertexColorStyle::Checkerboard};
    VertexColorMaterial debugCheckerMaterial{VertexColorStyle::DebugChecker};
    std::unique_ptr<Backend::Texture> materialTexture;
    std::unique_ptr<PhongMaterial> phong;
    std::unique_ptr<PBRMaterial> pbr;
    std::unique_ptr<PhongMaterial> skinnedPhong;
    std::unique_ptr<PBRMaterial> skinnedPbr;
    std::unique_ptr<Backend::Framebuffer> previewFramebuffer;
    std::unique_ptr<Backend::Framebuffer> rgba8PreviewFramebuffer;
    std::unique_ptr<Backend::Buffer> renderHookVertexBuffer;
    std::unique_ptr<Backend::GraphicsPipeline> renderHookPipeline;
    RenderHookHandle renderHook = InvalidRenderHook;
    int renderHookRecordCount = 0;
};

} // namespace

int main() {
    Phase1SceneSmoke app;
    app.initialize(320, 180, true, UpAxis::Y, Backend::BackendType::OpenGL,
                   Scene::BackendType::Native, true);
    app.start();
    require(app.completed, "phase 1 scene smoke did not finish");
    std::cout << "PASS: production Phase 1 scene renders Phong/PBR, alpha, "
                 "skinning, shadow, selection, debug/text, external instances, "
                 "repeated resize, and clean shutdown"
              << std::endl;
    return 0;
}

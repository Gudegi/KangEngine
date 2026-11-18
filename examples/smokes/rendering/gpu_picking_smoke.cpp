#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/native/native_scene.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
#include <cuda_runtime_api.h>
#endif

using namespace KE;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main() {
    Window window;
    window.init(128, 128, true);
    Backend::OpenGLDevice device;
    device.initialize();
    device.setValidationEnabled(true);
    Scene::NativeScene scene;
    ColorMaterial material;
    Rasterizer raster(&device);
    raster.setShadowDistance(0);
    const glm::mat4 view =
        glm::lookAt(glm::vec3(0, 0, 6), glm::vec3(0), glm::vec3(0, 1, 0));
    const glm::mat4 proj = glm::ortho(-4.0f, 4.0f, -4.0f, 4.0f, 0.1f, 20.0f);
    auto sync = [&] { raster.updateFrameData(view, proj); };
    auto pick = [&](float x, float y) {
        return raster.pickScreen({x / 4, y / 4}, view, proj, 128, 128);
    };
    auto define = [&](const char* path, std::shared_ptr<Scene::MeshData> mesh) {
        auto* prim = scene.definePrim(path, Scene::PrimType::Mesh);
        prim->setMeshData(std::move(mesh));
        return prim;
    };

    // The nearer AABB contains the click, but its triangle does not.
    auto largeMesh = std::make_shared<Scene::MeshData>();
    largeMesh->vertices = {{-3, -3, 1}, {3, -3, 1}, {3, 3, 1}};
    largeMesh->normals.assign(3, {0, 0, 1});
    largeMesh->indices = {0, 1, 2};
    auto* large = define("/large", largeMesh);
    raster.addRenderable(&material, large);
    auto cube =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(0.5f));
    auto* small = define("/small", cube);
    small->setLocalTranslation({-0.5f, 0.5f, 0});
    raster.addRenderable(&material, small);
    sync();
    const auto rayHit = raster.rayPick({{-0.5f, 0.5f, 5.9f}, {0, 0, -1}});
    require(rayHit.prim == large, "test did not reproduce the AABB overlap bug");
    auto hit = pick(-0.5f, 0.5f);
    require(hit.prim == small, "empty AABB space intercepted the surface pick");
    require(std::abs(hit.position.z - 0.25f) < 0.001f,
            "pick depth did not reconstruct the visible surface");
    require(!pick(-2, 2).hit, "empty background was picked");
    require(pick(1, 0).prim == large, "large mesh surface was not picked");
    large->setVisible(false);
    small->setVisible(false);

    // Bone movement leaves the bind-pose bounds, including unused bone slots.
    Scene::SkinnedMeshData skin;
    skin.mesh = *cube;
    skin.boneIndices.assign(skin.mesh.vertices.size(), {0, -1, -1, -1});
    skin.boneWeights.assign(skin.mesh.vertices.size(), {1, 0, 0, 0});
    auto* character =
        define("/character", std::make_shared<Scene::MeshData>(skin.mesh));
    const auto skinHandle =
        raster.addSkinnedRenderable(&material, character, skin);
    raster.updateRenderableSkinningMatrices(
        skinHandle, {glm::translate(glm::mat4(1), glm::vec3(2, 1, 0))});
    sync();
    require(!pick(0, 0).hit, "bind-pose bounds remained selectable");
    require(pick(2, 1).prim == character, "animated surface was not picked");
    require(character->getMeshData()->vertices == skin.mesh.vertices,
            "picking changed the source mesh");
    character->setVisible(false);

    auto* deform = define("/deform", std::make_shared<Scene::MeshData>(*cube));
    const auto deformHandle = raster.addRenderable(&material, deform);
    auto positions = cube->vertices;
    for (auto& position : positions)
        position += glm::vec3(-2, 1, 0);
    raster.updateRenderableGeometry(deformHandle, positions, cube->normals);
    sync();
    require(pick(-2, 1).prim == deform, "deformed GPU geometry was not picked");
    require(!pick(0, 0).hit, "undeformed source geometry remained selectable");
    deform->setVisible(false);

    // A non-pickable or fully transparent front instance must not hide a
    // target.
    auto* front = define("/front", cube);
    auto* back = define("/back", cube);
    front->setLocalTranslation({0, 0, 1});
    raster.addRenderable(&material, front);
    raster.addRenderable(&material, back);
    sync();
    require(pick(0, 0).prim == front,
            "depth test did not select the front surface");
    front->getSelectionComponent()->setPickable(false);
    require(pick(0, 0).prim == back,
            "non-pickable instance intercepted picking");
    front->getSelectionComponent()->setPickable(true);
    front->setDisplayColorAlpha({1, 1, 1, 0});
    sync();
    require(pick(0, 0).prim == back, "zero-alpha instance intercepted picking");
    front->setVisible(false);
    back->setVisible(false);

    // Alpha masks use texture alpha or a separate red-channel opacity map.
    const uint8_t texel[] = {0, 255, 255, 0};
    Backend::TextureResourceDesc textureDesc;
    textureDesc.extent = {1, 1, 1};
    textureDesc.format = Backend::TextureFormat::RGBA8Unorm;
    textureDesc.usage =
        Backend::TextureUsage::TextureBinding | Backend::TextureUsage::CopyDst;
    Backend::TextureInitialData initial{texel, sizeof(texel), sizeof(texel)};
    auto texture = device.createTexture(textureDesc, &initial);
    PhongMaterial maskedMaterial;
    maskedMaterial.setDiffuseMap(texture.get());
    auto* masked = define("/masked", cube);
    const auto maskedHandle = raster.addRenderable(&maskedMaterial, masked);
    raster.setRenderableAlphaMode(maskedHandle, AlphaMode::Mask, 0.5f);
    sync();
    require(!pick(0, 0).hit, "texture alpha cutout remained selectable");
    raster.setRenderableAlphaMode(maskedHandle, AlphaMode::Opaque);
    require(pick(0, 0).prim == masked,
            "opaque material incorrectly used texture alpha");
    raster.setRenderableAlphaMode(maskedHandle, AlphaMode::Mask, 0.5f);
    maskedMaterial.setAlphaMap(texture.get());
    require(!pick(0, 0).hit, "red-channel opacity cutout remained selectable");
    maskedMaterial.setAlphaMap(nullptr)->setDiffuseMap(nullptr);
    require(pick(0, 0).prim == masked, "opaque mask material was not picked");
    masked->setVisible(false);

    skin.boneIndices.assign(skin.mesh.vertices.size(), glm::ivec4(-1));
    skin.boneWeights.assign(skin.mesh.vertices.size(), glm::vec4(0));
    auto* maskedSkin =
        define("/masked_skin", std::make_shared<Scene::MeshData>(skin.mesh));
    const auto maskedSkinHandle =
        raster.addSkinnedRenderable(&maskedMaterial, maskedSkin, skin);
    raster.setRenderableAlphaMode(maskedSkinHandle, AlphaMode::Mask, 0.5f);
    maskedMaterial.setDiffuseMap(texture.get());
    sync();
    require(!pick(0, 0).hit, "skinned alpha cutout remained selectable");
    maskedMaterial.setDiffuseMap(nullptr);
    require(pick(0, 0).prim == maskedSkin,
            "zero-weight skin lost identity fallback");
    maskedSkin->setVisible(false);

    auto triangle = std::make_shared<Scene::MeshData>();
    triangle->vertices = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
    triangle->normals.assign(3, {0, 0, -1});
    triangle->indices = {2, 1, 0};
    auto* reversed = define("/reversed", triangle);
    const auto reversedHandle = raster.addRenderable(&material, reversed);
    sync();
    require(!pick(0, 0).hit, "back-face culling was ignored");
    raster.setRenderableDoubleSided(reversedHandle, true);
    require(pick(0, 0).prim == reversed, "double-sided surface was not picked");
    reversed->setVisible(false);

    auto* external = define("/external", cube);
    const auto externalHandle = raster.addRenderable(
        &material, external, TransformSource::ExternalBuffer);
    std::vector<glm::mat4> transforms(
        300, glm::translate(glm::mat4(1), glm::vec3(100, 0, 0)));
    transforms.back() = glm::translate(glm::mat4(1), glm::vec3(-2, -1, 0));
    raster.updateRenderableTransforms(externalHandle, transforms);
    sync();

    Backend::FramebufferDesc framebufferDesc{
        128, 128, false, true, 0, Backend::FramebufferColorFormat::RGBA16F};
    auto framebuffer = device.createFramebuffer(framebufferDesc);
    auto colorView =
        device.createTextureView(framebuffer->getColorTexture(), {});
    Backend::TextureResourceDesc multisample;
    multisample.extent = {128, 128, 1};
    multisample.sampleCount = 4;
    multisample.format = Backend::TextureFormat::RGBA16Float;
    multisample.usage = Backend::TextureUsage::RenderAttachment;
    auto color = device.createTexture(multisample);
    multisample.format = Backend::TextureFormat::Depth24Stencil8;
    auto depthTexture = device.createTexture(multisample);
    auto drawColorView = device.createTextureView(color.get(), {});
    auto depthView = device.createTextureView(depthTexture.get(), {});
    Backend::RenderPassDesc targetDesc;
    targetDesc.colorAttachments = {{drawColorView.get(),
                                    colorView.get(),
                                    Backend::LoadOp::Clear,
                                    Backend::StoreOp::Store,
                                    {0, 0, 0, 0}}};
    Backend::DepthStencilAttachmentDesc depth;
    depth.view = depthView.get();
    depth.depthLoadOp = Backend::LoadOp::Clear;
    targetDesc.depthStencilAttachment = depth;
    auto clearTarget = device.createRenderTarget(targetDesc);
    targetDesc.colorAttachments[0].loadOp = Backend::LoadOp::Load;
    targetDesc.depthStencilAttachment->depthLoadOp = Backend::LoadOp::Load;
    auto target = device.createRenderTarget(targetDesc);
    raster.setViewportSize(128, 128);
    auto render = [&] {
        auto encoder = device.createCommandEncoder();
        auto pass = encoder->beginRenderPass(clearTarget.get());
        pass->end();
        auto commands = encoder->finish();
        device.submit(*commands);
        raster.render(view, proj, target.get());
    };
    render();
    const auto before = device.readTexture(colorView.get());
    hit = pick(-2, -1);
    require(hit.handle == externalHandle && hit.instanceIndex == 299 &&
                !hit.prim,
            "frustum compaction changed the external instance identity");
    require(hit.transformSource == TransformSource::ExternalBuffer,
            "external transform ownership was lost");
    render();
    require(before.values == device.readTexture(colorView.get()).values,
            "picking changed normal render buffers or state");
    require(!raster.pickScreen({2, 0}, view, proj, 128, 128).hit,
            "out-of-viewport pick was accepted");
    require(raster.pickScreen({-0.5f, -0.25f}, view, proj, 257, 193).handle ==
                externalHandle,
            "resized viewport changed picking");

#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    auto checkCuda = [](cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    };
    void* cudaTransforms = nullptr;
    checkCuda(
        cudaMalloc(&cudaTransforms, transforms.size() * sizeof(glm::mat4)));
    auto releaseCuda = [](void* pointer) { cudaFree(pointer); };
    std::unique_ptr<void, decltype(releaseCuda)> allocation(cudaTransforms,
                                                            releaseCuda);
    checkCuda(cudaMemcpy(cudaTransforms, transforms.data(),
                         transforms.size() * sizeof(glm::mat4),
                         cudaMemcpyHostToDevice));
    Sim::GpuArrayView gpuView;
    gpuView.data = cudaTransforms;
    gpuView.memoryType = Sim::SimMemoryType::CUDADevice;
    gpuView.deviceId = 0;
    raster.setRenderableExternalBuffer(
        externalHandle, makeExternalMat4BufferDesc(gpuView, 300));
    sync();
    hit = pick(-2, -1);
    require(hit.handle == externalHandle && hit.instanceIndex == 299,
            "CUDA transform buffer was not picked");
    glm::mat4 unavailable;
    require(!raster.getRenderableInstanceTransform(externalHandle, 299,
                                                   unavailable),
            "GPU picking unexpectedly populated CPU transforms");
    auto mask = device.createFramebuffer({128, 128, false, false, 0});
    raster.renderSelectionMaskPass(hit, mask.get(), 128, 128);
    auto maskView = device.createTextureView(mask->getColorTexture(), {});
    const auto maskPixels = device.readTexture(maskView.get());
    bool foundMask = false;
    for (size_t i = 0; i < maskPixels.values.size(); i += 4)
        foundMask |= maskPixels.values[i] > 0.5f;
    require(foundMask, "CUDA instance selection mask is empty");
    raster.removePrim(externalHandle, external);
#endif
    require(glGetError() == GL_NO_ERROR,
            "GPU picking produced an OpenGL error");

    std::cout
        << "PASS: GPU surface picking, skinning, depth, alpha, visibility, "
           "instance identity, resize, and render preservation\n";
}

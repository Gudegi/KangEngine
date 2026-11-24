#pragma once

// Scene batches, dynamic updates, frame execution, and capture for renderers.

#include "engine/graphics/renderer/render_backend/render_image.hpp"
#include "engine/graphics/renderer/render_backend/render_picking.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include "engine/graphics/renderer/render_backend/render_debug_draw.hpp"
#include "utils/types.hpp"

#include <glm/mat4x4.hpp>
#include <stdexcept>
#include <optional>
#include <vector>

namespace KE {

class Camera;
namespace Scene {
struct RenderSceneFrame;
struct RenderLightData;
struct RenderMeshData;
struct RenderMaterialSource;
struct RenderInstanceData;
struct SkinnedMeshData;
} // namespace Scene

struct RenderBackendCapabilities {
    bool screenPresentation = false;
    bool selectionOutline = false;
    bool wireframe = false;
    bool staticScene = false;
    bool sceneBatches = false;
    // Texture creation from CPU pixel arrays.
    bool texturePixelsInput = false;
    bool screenPicking = false;
    // External GPU image output, currently used by ovrtx presentation.
    bool cudaImageOutput = false;
    // CPU transform arrays through the ExternalBuffer batch API.
    bool cpuInstanceTransformsInput = false;
    bool skinning = false;
};

struct SceneBatchState {
    bool doubleSided = false;
    bool castsShadow = true;
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;

    bool operator==(const SceneBatchState& other) const {
        return doubleSided == other.doubleSided &&
               castsShadow == other.castsShadow &&
               alphaMode == other.alphaMode && alphaCutoff == other.alphaCutoff;
    }
    bool operator!=(const SceneBatchState& other) const {
        return !(*this == other);
    }
};

class RenderBackend {
  public:
    virtual ~RenderBackend() = default;
    // Fixed for the initialized session.
    virtual RenderBackendCapabilities capabilities() const = 0;
    virtual void initialize(int width, int height) = 0;
    virtual void resize(int width, int height) = 0;
    virtual void setBackgroundColor(const glm::vec4&) {
        throw std::runtime_error(
            "Render backend does not support background color");
    }
    virtual void setGroundStyle(const BackgroundSettings&) {
        throw std::runtime_error(
            "Render backend does not support ground styling");
    }
    virtual void setExposure(float) {
        throw std::runtime_error("Render backend does not support exposure");
    }
    virtual RenderDebugDraw& debugDraw() {
        throw std::runtime_error("Render backend does not support debug drawing");
    }
    // Full state or explicit upserts/removals from one resource manager; no
    // Prim access. Unchanged lighting is retained. Camera matrices still come
    // from frame calls.
    virtual void syncScene(const Scene::RenderSceneFrame& scene) = 0;
    virtual void syncLights(const Scene::RenderLightData&) {
        throw std::runtime_error(
            "Render backend does not support scene lights");
    }
    // Borrowed arrays/material source stay alive until batch removal. No
    // Prim access. ExternalBuffer updates override owner poses; visibility is
    // batch-wide. Optional skin weights are consumed during registration.
    virtual RenderableHandle
    registerSceneBatch(const Scene::RenderMeshData& mesh,
                       const Scene::RenderMaterialSource& material,
                       const Scene::RenderInstanceData& instances,
                       TransformSource source = TransformSource::SceneGraph,
                       const Scene::SkinnedMeshData* skinnedMesh = nullptr) = 0;
    virtual void removeSceneBatch(RenderableHandle) {
        throw std::runtime_error("Render backend does not support batch removal");
    }
    virtual void setSceneBatchState(RenderableHandle handle,
                                    const SceneBatchState& state) = 0;
    // CPU arrays are consumed during the call; source MeshData stays immutable.
    virtual void updateRenderableTransforms(RenderableHandle,
                                            const std::vector<glm::mat4>&,
                                            const std::vector<glm::vec4>*) {
        throw std::runtime_error(
            "Render backend does not support instance updates");
    }
    // Borrowed memory retains its device, lifetime, and synchronization
    // contract.
    virtual void setRenderableExternalBuffer(RenderableHandle,
                                             const ExternalBufferDesc&) {
        throw std::runtime_error(
            "Render backend does not support external buffers");
    }
    virtual bool getRenderableInstanceTransform(RenderableHandle, int,
                                                glm::mat4&) const {
        throw std::runtime_error(
            "Render backend does not support instance queries");
    }
    virtual bool setRenderableInstanceTransform(RenderableHandle, int,
                                                const glm::mat4&) {
        throw std::runtime_error(
            "Render backend does not support instance editing");
    }
    virtual void setRenderableColors(RenderableHandle,
                                     const std::vector<glm::vec4>&) {
        throw std::runtime_error(
            "Render backend does not support instance colors");
    }
    // Positions/normals are mesh-local; empty normals retain existing normals.
    virtual void updateRenderableGeometry(RenderableHandle,
                                          const std::vector<glm::vec3>&,
                                          const std::vector<glm::vec3>&) {
        throw std::runtime_error(
            "Render backend does not support geometry updates");
    }
    virtual void
    updateRenderableSkinningMatrices(RenderableHandle,
                                     const std::vector<glm::mat4>&) {
        throw std::runtime_error(
            "Render backend does not support skinning updates");
    }
    // GLM column-major matrices; scene synchronization precedes interaction.
    virtual void syncFrame(const glm::mat4& view,
                           const glm::mat4& projection) = 0;
    virtual void prepareFrame(Camera& camera, UpAxis upAxis) = 0;
    virtual void beginFrame() = 0;
    virtual void renderScene(const glm::mat4& view, const glm::mat4& projection,
                             bool wireframe) = 0;
    // Latest request supersedes earlier work. Polling never advances simulation
    // or renders another frame; immediate completion is allowed.
    virtual void requestPick(const RenderPickRequest&) {
        throw std::runtime_error(
            "Render backend does not support screen picking");
    }
    virtual std::optional<RenderPickResult> pollPick() { return std::nullopt; }
    virtual void endFrame(const RenderPickHit& selection,
                          bool presentToScreen) = 0;
    // Synchronous owned RGB8; flipY=true is top-down, false is bottom-up.
    virtual RenderImage readImage(bool flipY) = 0;
    virtual RenderImage readImageResized(int width, int height, bool flipY) = 0;
    // Borrowed GPU output; Native exposes its texture through Renderer instead.
    virtual const CudaRenderImage* cudaImage() const { return nullptr; }
};

} // namespace KE

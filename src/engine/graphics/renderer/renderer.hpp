#ifndef _RENDERER_HPP_
#define _RENDERER_HPP_

#include "engine/graphics/renderer/light.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>

namespace KE {

class Camera;
class Rasterizer;
class PostProcessor;
class SelectionOutlineProcessor;

namespace Backend {
class Framebuffer;
class GraphicsDevice;
class Texture;
} // namespace Backend

// Facade for render-system access. App owns the concrete resources; Renderer
// only groups the public rendering surface so App does not expose every member
// directly as the primary API.
class Renderer {
  public:
    void bind(Backend::GraphicsDevice* device, Rasterizer* rasterizer,
              PostProcessor* postProcessor,
              SelectionOutlineProcessor* selectionOutlineProcessor);
    void setViewportSize(int width, int height);

    Backend::GraphicsDevice* device() { return _device; }
    const Backend::GraphicsDevice* device() const { return _device; }
    Rasterizer* rasterizer() { return _rasterizer; }
    const Rasterizer* rasterizer() const { return _rasterizer; }
    PostProcessor* postProcessor() { return _postProcessor; }
    const PostProcessor* postProcessor() const { return _postProcessor; }
    SelectionOutlineProcessor* selectionOutline() {
        return _selectionOutlineProcessor;
    }
    const SelectionOutlineProcessor* selectionOutline() const {
        return _selectionOutlineProcessor;
    }

    void setLight(const DirectionalLight& light);
    const DirectionalLight& light() const;
    Backend::Framebuffer* shadowFbo();
    void renderSceneToFramebuffer(Camera& camera, Backend::Framebuffer* target,
                                  int width, int height, bool clear = true);

    // MeshHandle identifies a renderable batch/instancer. Most controls apply
    // to the whole batch; APIs with instanceIndex can target one instance.
    void
    updateRenderableTransforms(MeshHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors = nullptr);
    bool getRenderableInstanceTransform(MeshHandle handle, int instanceIndex,
                                        glm::mat4& outTransform) const;
    bool setRenderableInstanceTransform(MeshHandle handle, int instanceIndex,
                                        const glm::mat4& transform);
    void setRenderableColors(MeshHandle handle,
                             const std::vector<glm::vec4>& colors);
    void setRenderableDoubleSided(MeshHandle handle, bool doubleSided = true);
    void setRenderableCastsShadow(MeshHandle handle, bool castsShadow = true);
    void setRenderableTexture(MeshHandle handle, Backend::Texture* tex,
                              int slot = 0);
    void updateRenderableGeometry(MeshHandle handle,
                                  const std::vector<glm::vec3>& positions,
                                  const std::vector<glm::vec3>& normals);
    void updateRenderableSkinningMatrices(
        MeshHandle handle, const std::vector<glm::mat4>& boneMatrices);

  private:
    Backend::GraphicsDevice* _device = nullptr;
    Rasterizer* _rasterizer = nullptr;
    PostProcessor* _postProcessor = nullptr;
    SelectionOutlineProcessor* _selectionOutlineProcessor = nullptr;
    int _viewportWidth = 0;
    int _viewportHeight = 0;
};

} // namespace KE

#endif

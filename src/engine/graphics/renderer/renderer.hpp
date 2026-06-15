#ifndef _RENDERER_HPP_
#define _RENDERER_HPP_

#include "engine/graphics/renderer/light.hpp"

namespace KE {

class Camera;
class Rasterizer;
class PostProcessor;
class SelectionOutlineProcessor;

namespace Backend {
class Framebuffer;
class GraphicsDevice;
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

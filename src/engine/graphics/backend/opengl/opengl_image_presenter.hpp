#pragma once

// Presents external render output (currently ovrtx) through CUDA -> OpenGL.

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/renderer/render_backend/render_image.hpp"

struct cudaGraphicsResource;

namespace KE::Backend {

class OpenGLImagePresenter {
  public:
    explicit OpenGLImagePresenter(GraphicsDevice& device);
    ~OpenGLImagePresenter();
    OpenGLImagePresenter(const OpenGLImagePresenter&) = delete;
    OpenGLImagePresenter& operator=(const OpenGLImagePresenter&) = delete;

    // Waits for producer readiness and completes the copy before returning.
    void upload(const CudaRenderImage& image);
    // Full-window output; the editor viewport samples texture() instead.
    void present(int width, int height);
    Texture* texture() const { return _texture.get(); }

  private:
    GraphicsDevice& _device;
    std::unique_ptr<Texture> _texture;
    unsigned int _framebuffer = 0;
    cudaGraphicsResource* _cudaResource = nullptr;
    int _cudaDevice = -1;
};

} // namespace KE::Backend

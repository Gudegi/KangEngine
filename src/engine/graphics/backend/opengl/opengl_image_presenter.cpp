#include "opengl_image_presenter.hpp"

#include <glad/glad.h>
#include <stdexcept>
#include <string>

#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#endif

namespace KE::Backend {
namespace {
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

class CudaDeviceScope {
  public:
    explicit CudaDeviceScope(int device) {
        checkCuda(cudaGetDevice(&_previous), "cudaGetDevice");
        checkCuda(cudaSetDevice(device), "cudaSetDevice");
    }
    ~CudaDeviceScope() { cudaSetDevice(_previous); }

  private:
    int _previous = 0;
};
#endif
} // namespace

OpenGLImagePresenter::OpenGLImagePresenter(GraphicsDevice& device)
    : _device(device) {
    if (device.getBackendType() != BackendType::OpenGL)
        throw std::invalid_argument("CUDA image presentation requires OpenGL");
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    unsigned int count = 0;
    checkCuda(
        cudaGLGetDevices(&count, &_cudaDevice, 1, cudaGLDeviceListCurrentFrame),
        "cudaGLGetDevices");
    if (count != 1)
        throw std::runtime_error(
            "Presentation requires one CUDA-capable GL device");
    glGenFramebuffers(1, &_framebuffer);
#else
    throw std::runtime_error(
        "GPU image presentation requires USE_CUDA_INTEROP");
#endif
}

OpenGLImagePresenter::~OpenGLImagePresenter() {
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    if (_cudaResource) {
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(_cudaDevice);
        cudaGraphicsUnregisterResource(_cudaResource);
        cudaSetDevice(previous);
    }
#endif
    glDeleteFramebuffers(1, &_framebuffer);
}

void OpenGLImagePresenter::upload(const CudaRenderImage& image) {
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    if (image.width <= 0 || image.height <= 0 || !image.array)
        throw std::invalid_argument(
            "Presentation requires a nonempty CUDA image");
    if (image.device != _cudaDevice)
        throw std::invalid_argument(
            "Render image and OpenGL must use the same GPU");
    CudaDeviceScope device(_cudaDevice);
    auto source = reinterpret_cast<cudaArray_t>(image.array);

    // Host waiting avoids the SDK's documented Linux Vulkan/stream-wait
    // scheduling interaction. No pixel data crosses to the CPU.
    if (image.readyEvent) {
        auto scope = _device.profileContext()->cpuScope("presentation/wait");
        checkCuda(cudaEventSynchronize(
                      reinterpret_cast<cudaEvent_t>(image.readyEvent)),
                  "render image readiness");
    }
    auto scope = _device.profileContext()->cpuScope("presentation/gpu_copy");
    cudaChannelFormatDesc format;
    cudaExtent extent;
    unsigned int flags;
    checkCuda(cudaArrayGetInfo(&format, &extent, &flags, source),
              "cudaArrayGetInfo");
    if (format.f != cudaChannelFormatKindUnsigned || format.x != 8 ||
        format.y != 8 || format.z != 8 || format.w != 8 ||
        extent.width != static_cast<size_t>(image.width) ||
        extent.height != static_cast<size_t>(image.height))
        throw std::invalid_argument(
            "Presentation requires a matching RGBA8 CUDA array");

    // Allocate the OpenGL destination for the external renderer's image.
    if (!_texture || _texture->getWidth() != image.width ||
        _texture->getHeight() != image.height) {
        if (_cudaResource) {
            checkCuda(cudaGraphicsUnregisterResource(_cudaResource),
                      "cudaGraphicsUnregisterResource");
            _cudaResource = nullptr;
        }
        TextureResourceDesc desc;
        desc.extent = {static_cast<uint32_t>(image.width),
                       static_cast<uint32_t>(image.height)};
        desc.format = TextureFormat::RGBA8Unorm;
        desc.usage = TextureUsage::TextureBinding |
                     TextureUsage::RenderAttachment | TextureUsage::CopyDst;
        desc.label = "render_image_presentation";
        _texture = _device.createTexture(desc);
        GLint previous = 0;
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previous);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _framebuffer);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, _texture->getNativeHandle(), 0);
        const auto status = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previous);
        if (status != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error(
                "Image presentation framebuffer incomplete");
    }
    if (!_cudaResource)
        checkCuda(cudaGraphicsGLRegisterImage(
                      &_cudaResource, _texture->getNativeHandle(),
                      GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard),
                  "cudaGraphicsGLRegisterImage");

    // Copy the borrowed ovrtx CUDA image into the shared OpenGL texture.
    checkCuda(cudaGraphicsMapResources(1, &_cudaResource),
              "cudaGraphicsMapResources");
    try {
        cudaArray_t destination;
        checkCuda(cudaGraphicsSubResourceGetMappedArray(&destination,
                                                        _cudaResource, 0, 0),
                  "cudaGraphicsSubResourceGetMappedArray");
        checkCuda(cudaMemcpy2DArrayToArray(destination, 0, 0, source, 0, 0,
                                           static_cast<size_t>(image.width) * 4,
                                           image.height,
                                           cudaMemcpyDeviceToDevice),
                  "render image GPU copy");
    } catch (...) {
        cudaGraphicsUnmapResources(1, &_cudaResource);
        cudaStreamSynchronize(nullptr);
        throw;
    }
    checkCuda(cudaGraphicsUnmapResources(1, &_cudaResource),
              "cudaGraphicsUnmapResources");
    checkCuda(cudaStreamSynchronize(nullptr), "render image copy completion");
#else
    throw std::runtime_error(
        "GPU image presentation requires USE_CUDA_INTEROP");
#endif
}

void OpenGLImagePresenter::present(int width, int height) {
    if (!_texture)
        return;
    // Present the copied image with top-down rows and its existing gamma.
    const bool srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
    const bool scissor = glIsEnabled(GL_SCISSOR_TEST);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_SCISSOR_TEST);
    GLint read = 0, draw = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _framebuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, _texture->getHeight(), _texture->getWidth(), 0, 0, 0,
                      width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, read);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw);
    if (srgb)
        glEnable(GL_FRAMEBUFFER_SRGB);
    if (scissor)
        glEnable(GL_SCISSOR_TEST);
}

} // namespace KE::Backend

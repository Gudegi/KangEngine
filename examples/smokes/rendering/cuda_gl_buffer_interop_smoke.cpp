#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace KE;
using namespace KE::Backend;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}
} // namespace

int main() {
    Window window;
    window.init(32, 32, true);
    require(window.getGlfwWindow() != nullptr, "hidden window creation failed");

    OpenGLDevice device;
    device.initialize();
    device.setValidationEnabled(true);

    constexpr std::array<float, 8> expected = {
        1.0f, -2.0f, 3.5f, 4.25f, 8.0f, 13.0f, 21.0f, 34.0f};
    float* cudaValues = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&cudaValues),
                         sizeof(expected)),
              "cudaMalloc");

    try {
        checkCuda(cudaMemcpy(cudaValues, expected.data(), sizeof(expected),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy host to device");

        BufferDesc desc;
        desc.size = sizeof(expected);
        desc.usage = BufferUsage::Vertex | BufferUsage::CopyDst;
        desc.label = "cuda_gl_interop_smoke_buffer";
        auto buffer = device.createBuffer(desc);

        Sim::GpuArrayView source;
        source.data = cudaValues;
        source.memoryType = Sim::SimMemoryType::CUDADevice;
        source.dtype = Sim::SimDType::Float32;
        source.deviceId = 0;
        source.shape = {static_cast<int64_t>(expected.size())};
        source.name = "cuda_gl_interop_smoke_source";
        require(buffer->setExternalData(source, expected.size(), sizeof(float),
                                        sizeof(float)),
                "CUDA external upload was rejected");

        auto* glBuffer = dynamic_cast<OpenGLBuffer*>(buffer.get());
        require(glBuffer != nullptr, "expected OpenGL buffer implementation");
        std::array<float, expected.size()> actual{};
        glBindBuffer(GL_ARRAY_BUFFER, glBuffer->getHandle());
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(actual), actual.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        require(glGetError() == GL_NO_ERROR, "OpenGL buffer readback failed");
        for (size_t i = 0; i < expected.size(); ++i)
            require(std::abs(actual[i] - expected[i]) < 1.0e-6f,
                    "CUDA/OpenGL interop produced incorrect data");

        buffer.reset();
        checkCuda(cudaFree(cudaValues), "cudaFree");
        std::cout << "PASS: CUDA device data copied through OpenGL external "
                     "buffer interop"
                  << std::endl;
        return 0;
    } catch (...) {
        cudaFree(cudaValues);
        throw;
    }
}
